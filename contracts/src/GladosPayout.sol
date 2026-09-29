// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// One epoch's payout in one call: buy GLADOS with ETH, split it equally.
///
/// **Atomic, because the alternative leaks.** Buying through a V2 pair without
/// a router is three steps -- wrap, send WETH to the pair, call `swap` -- and
/// done as separate transactions the WETH sits in the pair between them, where
/// `skim` or anyone's `swap` can take it. Here all three happen inside one call,
/// then the split, so there is no moment when the pool's money is anywhere but
/// on its way to the miners.
///
/// **Measured, not predicted.** GLADOS takes a buy tax out of the pair's
/// transfer, so what arrives is less than the swap's `out`, by a rate that is
/// the token's to change. The split divides what this contract's balance
/// actually rose by, and each recipient's balance must then rise by exactly
/// their share -- `design/audit-2.md` F4, applied to the send side -- or the
/// whole payout reverts rather than shorting a miner silently.
///
/// **A slippage bound is required.** `minOut` comes from the caller's quote a
/// moment earlier; zero is refused, because a swap with no bound is one a
/// sandwich can take all of.
///
/// **Holds nothing.** Every token that arrives in a call leaves in it: the equal
/// shares to the miners and the rounding dust back to the caller, so miners
/// stay exactly equal and there is no balance here to be stuck or taken. No
/// owner, no state beyond the three immutables.
contract GladosPayout {
    address public immutable weth;
    address public immutable token;
    address public immutable pair;
    bool public immutable wethIsToken0;

    error BadPair();
    error NoRecipients();
    error NoSlippageBound();
    error TooLittleOut(uint256 got, uint256 minOut);
    error TransferFailed();
    error ShortPaid(uint256 index, uint256 wanted, uint256 got);

    event Paid(uint256 ethIn, uint256 tokenOut, uint256 recipients, uint256 each);

    constructor(address weth_, address token_, address pair_) {
        // Checked against the pair's own answer, as the distributor does: a
        // pair cannot be derived from two tokens without trusting a factory,
        // but it can be checked against them.
        address t0 = IPair(pair_).token0();
        address t1 = IPair(pair_).token1();
        if (!((t0 == weth_ && t1 == token_) || (t0 == token_ && t1 == weth_))) revert BadPair();
        weth = weth_;
        token = token_;
        pair = pair_;
        wethIsToken0 = t0 == weth_;
    }

    function buyAndPay(address[] calldata to, uint256 minOut) external payable returns (uint256 each) {
        if (to.length == 0) revert NoRecipients();
        if (minOut == 0) revert NoSlippageBound();
        uint256 got = _buy(msg.value);
        if (got < minOut) revert TooLittleOut(got, minOut);
        each = _pay(to, got);
        emit Paid(msg.value, got, to.length, each);
    }

    /// Wrap, send to the pair, swap to this contract. Answers what arrived.
    function _buy(uint256 amount) private returns (uint256 got) {
        IWETH(weth).deposit{value: amount}();
        _move(weth, abi.encodeWithSelector(0xa9059cbb, pair, amount));
        (uint112 r0, uint112 r1,) = IPair(pair).getReserves();
        (uint256 rIn, uint256 rOut) = wethIsToken0 ? (uint256(r0), uint256(r1)) : (uint256(r1), uint256(r0));
        uint256 out = _amountOut(amount, rIn, rOut);
        uint256 before = _balanceOf(token, address(this));
        if (wethIsToken0) IPair(pair).swap(0, out, address(this), "");
        else IPair(pair).swap(out, 0, address(this), "");
        got = _balanceOf(token, address(this)) - before;
    }

    /// Equal shares, each measured on arrival; the dust back to the caller.
    function _pay(address[] calldata to, uint256 got) private returns (uint256 each) {
        each = got / to.length;
        for (uint256 i = 0; i < to.length; ++i) {
            uint256 b = _balanceOf(token, to[i]);
            _move(token, abi.encodeWithSelector(0xa9059cbb, to[i], each));
            uint256 rose = _balanceOf(token, to[i]) - b;
            if (rose != each) revert ShortPaid(i, each, rose);
        }
        uint256 dust = got - each * to.length;
        if (dust > 0) _move(token, abi.encodeWithSelector(0xa9059cbb, msg.sender, dust));
    }

    /// Uniswap V2's constant-product formula with its 0.3% fee -- the
    /// distributor's `_amountOut`, the same arithmetic the pair's `k` check holds
    /// it to.
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
