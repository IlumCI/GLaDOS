// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// One epoch's payout in one call, in whatever each miner chose to be paid in.
///
/// A miner chooses $GLADOS (the default), one tokenized stock, or a basket of
/// them (pool/edge/worker/rewards.js). The treasury groups the epoch's
/// recipients by choice, gives every wallet the same share of the pot, and
/// calls this once: each group's share is spent on its legs and each leg's
/// output is split equally between that group's wallets.
///
/// Two ways to buy, decided by the leg's token:
///
/// - **$GLADOS** on its Uniswap V2 pair, exactly as `GladosPayout` did: send
///   WETH to the pair, swap, measure what arrived after the buy tax.
/// - **A stock token** in two Uniswap V3 hops, WETH to USDG on the chain's
///   WETH/USDG pool and USDG to the stock on its own pool, because that is
///   where stock liquidity is (design/rwa.md: $8.4M on V3 against $61 on V2).
///
/// Everything `GladosPayout` promised still holds:
///
/// - **Atomic.** The whole epoch, every group and every leg, happens in one
///   call, so the pool's money is never resting where a third party can reach
///   it, and either every miner is paid or nobody is.
/// - **Measured.** Every hop is measured by this contract's balance change, and
///   every recipient's balance must rise by exactly their share, or the call
///   reverts rather than shorting somebody (`design/audit-2.md` F4).
/// - **Bounded.** Every leg carries `minOut` from the caller's quote; zero is
///   refused, because a swap with no bound is one a sandwich can take all of.
/// - **Holds nothing.** Rounding dust goes back to the caller.
///
/// The V3 callback is the one function here that pays out on request, and it
/// is guarded as the distributor's is: only the pool this contract is swapping
/// on, for exactly the amount it set out to sell, in the token it is selling,
/// once. The factory is used only to check that a pool is real before swapping
/// on it; authorising the callback by factory lookup is how it gets drained.
contract GladosPayout2 {
    address public immutable weth;
    address public immutable glados;
    address public immutable pair;
    bool public immutable wethIsToken0;
    address public immutable usdg;
    address public immutable factory;
    address public immutable wethUsdgPool;

    struct Leg {
        address pool;   // the V2 pair for $GLADOS, the token's USDG pool otherwise
        address token;  // what this leg buys
        uint256 eth;    // how much of msg.value it spends
        uint256 minOut; // the least of `token` it may receive
    }

    struct Group {
        Leg[] legs;
        address[] to;
    }

    address private _inFlight;
    address private _payToken;
    uint256 private _budget;

    uint160 private constant MIN_SQRT_RATIO = 4295128739;
    uint160 private constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;

    error BadPair();
    error BadPool();
    error BadCallback();
    error NoGroups();
    error NoLegs();
    error NoRecipients();
    error NoSlippageBound();
    error ValueMismatch();
    error TooLittleOut(uint256 group, uint256 leg, uint256 got, uint256 minOut);
    error TransferFailed();
    error ShortPaid(address token, uint256 index, uint256 wanted, uint256 got);

    event Paid(address indexed token, uint256 ethIn, uint256 tokenOut, uint256 recipients, uint256 each);

    constructor(address weth_, address glados_, address pair_, address usdg_, address factory_, address wethUsdgPool_) {
        address t0 = IPair(pair_).token0();
        address t1 = IPair(pair_).token1();
        if (!((t0 == weth_ && t1 == glados_) || (t0 == glados_ && t1 == weth_))) revert BadPair();
        weth = weth_;
        glados = glados_;
        pair = pair_;
        wethIsToken0 = t0 == weth_;
        usdg = usdg_;
        factory = factory_;
        _checkPool(wethUsdgPool_, weth_, usdg_);
        wethUsdgPool = wethUsdgPool_;
    }

    function payAll(Group[] calldata groups) external payable {
        if (groups.length == 0) revert NoGroups();
        uint256 total;
        for (uint256 g = 0; g < groups.length; ++g) {
            if (groups[g].legs.length == 0) revert NoLegs();
            if (groups[g].to.length == 0) revert NoRecipients();
            for (uint256 i = 0; i < groups[g].legs.length; ++i) {
                Leg calldata l = groups[g].legs[i];
                if (l.minOut == 0) revert NoSlippageBound();
                if (l.token == glados) {
                    if (l.pool != pair) revert BadPair();
                } else {
                    _checkPool(l.pool, usdg, l.token);
                }
                total += l.eth;
            }
        }
        if (total != msg.value || total == 0) revert ValueMismatch();

        IWETH(weth).deposit{value: msg.value}();
        for (uint256 g = 0; g < groups.length; ++g) {
            for (uint256 i = 0; i < groups[g].legs.length; ++i) {
                Leg calldata l = groups[g].legs[i];
                uint256 got = l.token == glados
                    ? _buyV2(l.eth)
                    : _swap(l.pool, usdg, l.token, _swap(wethUsdgPool, weth, usdg, l.eth));
                if (got < l.minOut) revert TooLittleOut(g, i, got, l.minOut);
                uint256 each = _pay(l.token, groups[g].to, got);
                emit Paid(l.token, l.eth, got, groups[g].to.length, each);
            }
        }
    }

    /// A V3 pool is believed only if the factory names it for exactly these two
    /// tokens at its own fee.
    function _checkPool(address pool, address a, address b) private view {
        address t0 = IUniswapV3Pool(pool).token0();
        address t1 = IUniswapV3Pool(pool).token1();
        if (!((t0 == a && t1 == b) || (t0 == b && t1 == a))) revert BadPool();
        if (IUniswapV3Factory(factory).getPool(t0, t1, IUniswapV3Pool(pool).fee()) != pool) revert BadPool();
    }

    /// WETH to the pair, swap to this contract; answers what arrived.
    function _buyV2(uint256 amount) private returns (uint256 got) {
        _move(weth, abi.encodeWithSelector(0xa9059cbb, pair, amount));
        (uint112 r0, uint112 r1,) = IPair(pair).getReserves();
        (uint256 rIn, uint256 rOut) = wethIsToken0 ? (uint256(r0), uint256(r1)) : (uint256(r1), uint256(r0));
        uint256 out = _amountOut(amount, rIn, rOut);
        uint256 before = _balanceOf(glados, address(this));
        if (wethIsToken0) IPair(pair).swap(0, out, address(this), "");
        else IPair(pair).swap(out, 0, address(this), "");
        got = _balanceOf(glados, address(this)) - before;
    }

    /// Exact-input V3 swap of `amount` of `sell` for `buy` on `pool`, to this
    /// contract; answers what arrived, measured.
    function _swap(address pool, address sell, address buy, uint256 amount) private returns (uint256 got) {
        bool zeroForOne = sell < buy;
        uint256 before = _balanceOf(buy, address(this));
        (address prevPool, address prevToken, uint256 prevBudget) = (_inFlight, _payToken, _budget);
        _inFlight = pool;
        _payToken = sell;
        _budget = amount;
        IUniswapV3Pool(pool).swap(
            address(this), zeroForOne, int256(amount), zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1, ""
        );
        // The callback spends the budget to zero or refuses, so any left means
        // the pool never asked to be paid and this was not a real swap.
        if (_budget != 0) revert BadCallback();
        (_inFlight, _payToken, _budget) = (prevPool, prevToken, prevBudget);
        got = _balanceOf(buy, address(this)) - before;
    }

    /// Pay the pool this contract is in the middle of swapping on: exactly the
    /// amount it set out to sell, in the token it is selling, once.
    function uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes calldata) external {
        if (msg.sender != _inFlight || _inFlight == address(0)) revert BadCallback();
        int256 owed = amount0Delta > 0 ? amount0Delta : amount1Delta;
        if (owed <= 0 || uint256(owed) != _budget) revert BadCallback();
        _budget = 0;
        address t = _payToken;
        _inFlight = address(0);
        _move(t, abi.encodeWithSelector(0xa9059cbb, msg.sender, uint256(owed)));
    }

    /// Equal shares of `token`, each measured on arrival; dust to the caller.
    function _pay(address token, address[] calldata to, uint256 got) private returns (uint256 each) {
        each = got / to.length;
        for (uint256 i = 0; i < to.length; ++i) {
            uint256 b = _balanceOf(token, to[i]);
            _move(token, abi.encodeWithSelector(0xa9059cbb, to[i], each));
            uint256 rose = _balanceOf(token, to[i]) - b;
            if (rose != each) revert ShortPaid(token, i, each, rose);
        }
        uint256 dust = got - each * to.length;
        if (dust > 0) _move(token, abi.encodeWithSelector(0xa9059cbb, msg.sender, dust));
    }

    /// Uniswap V2's constant-product formula with its 0.3% fee.
    function _amountOut(uint256 amountIn, uint256 rIn, uint256 rOut) private pure returns (uint256) {
        if (amountIn == 0 || rIn == 0 || rOut == 0) return 0;
        uint256 withFee = amountIn * 997;
        return (withFee * rOut) / (rIn * 1000 + withFee);
    }

    /// Empty return data is success; a returned word must be exactly 1 (F7).
    function _move(address asset, bytes memory data) private {
        (bool ok, bytes memory ret) = asset.call(data);
        if (!ok || (ret.length != 0 && (ret.length < 32 || abi.decode(ret, (uint256)) != 1))) revert TransferFailed();
    }

    function _balanceOf(address asset, address who) private view returns (uint256) {
        (bool ok, bytes memory ret) = asset.staticcall(abi.encodeWithSelector(0x70a08231, who));
        if (!ok || ret.length < 32) revert TransferFailed();
        return abi.decode(ret, (uint256));
    }
}

interface IWETH {
    function deposit() external payable;
}

interface IPair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112, uint112, uint32);
    function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data) external;
}

interface IUniswapV3Pool {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function fee() external view returns (uint24);
    function swap(address recipient, bool zeroForOne, int256 amountSpecified, uint160 sqrtPriceLimitX96, bytes calldata data)
        external
        returns (int256, int256);
}

interface IUniswapV3Factory {
    function getPool(address a, address b, uint24 fee) external view returns (address);
}
