// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/// Fixtures for `test/audit2.mjs`, the second, independent review of
/// `GladosDistributor`. Test-only: nothing here is deployed, and none of it is
/// compiled by `test/build.mjs`. Each contract is the smallest thing that makes
/// one suspicion from `design/audit-2.md` expressible as a claim.

interface IERC20Min {
    function balanceOf(address) external view returns (uint256);
    function transfer(address, uint256) external returns (bool);
    function transferFrom(address, address, uint256) external returns (bool);
    function approve(address, uint256) external returns (bool);
}

interface ICallback {
    function uniswapV3SwapCallback(int256, int256, bytes calldata) external;
}

/// An ERC-20 whose return convention and fee model are switchable, so `_move`
/// can be fed every shape of answer a token can give.
contract OddToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    uint256 public totalSupply;

    /// 0 normal (true), 1 explicit false, 2 one byte, 3 a 32-byte word of 2,
    /// 4 64 bytes whose first word is 1, 5 return nothing AND move nothing.
    uint8 public mode;
    /// Charged to the *sender* on top of the amount, which is the fee model
    /// `GladosDistributor` does not measure on its outbound legs.
    uint256 public senderFeeBps;
    address public feeSink = address(0xFEE);

    constructor(uint256 supply) {
        balanceOf[msg.sender] = supply;
        totalSupply = supply;
    }

    function setMode(uint8 m) external { mode = m; }
    function setSenderFee(uint256 bps) external { senderFeeBps = bps; }
    function mint(address to, uint256 a) external { balanceOf[to] += a; totalSupply += a; }

    function approve(address s, uint256 a) external returns (bool) {
        allowance[msg.sender][s] = a;
        return true;
    }

    function transfer(address to, uint256 a) external returns (bool) {
        return _do(msg.sender, to, a);
    }

    function transferFrom(address f, address to, uint256 a) external returns (bool) {
        require(allowance[f][msg.sender] >= a, "allowance");
        allowance[f][msg.sender] -= a;
        return _do(f, to, a);
    }

    function _do(address f, address to, uint256 a) private returns (bool) {
        if (mode == 5) {
            assembly { return(0, 0) }
        }
        uint256 fee = (a * senderFeeBps) / 10_000;
        require(balanceOf[f] >= a + fee, "balance");
        balanceOf[f] -= a + fee;
        balanceOf[to] += a;
        balanceOf[feeSink] += fee;
        if (mode == 1) return false;
        if (mode == 2) {
            assembly { mstore(0, shl(248, 1)) return(0, 1) }
        }
        if (mode == 3) {
            assembly { mstore(0, 2) return(0, 32) }
        }
        if (mode == 4) {
            assembly { mstore(0, 1) mstore(32, 0xdeadbeef) return(0, 64) }
        }
        return true;
    }
}

/// A pool the (mock) factory vouches for, whose behaviour is not a genuine
/// pool's. Used to price what the callback trusts about a vouched pool: how
/// much it asks for, and how many times.
contract EvilV3Pool {
    address public immutable token0;
    address public immutable token1;
    uint24 public immutable fee;

    /// What to ask for, as a fraction of `amountSpecified`, in bps. 10000 is
    /// honest; below it is a partial fill; above it is an over-ask.
    uint256 public askBps = 10_000;
    /// How many times to call back per swap.
    uint256 public calls = 1;
    /// Reward paid out per unit of input, 1:1 by default.
    constructor(address a, address b, uint24 fee_) {
        (token0, token1) = a < b ? (a, b) : (b, a);
        fee = fee_;
    }

    function set(uint256 askBps_, uint256 calls_) external {
        askBps = askBps_;
        calls = calls_;
    }

    function swap(address recipient, bool zeroForOne, int256 amountSpecified, uint160, bytes calldata data)
        external
        returns (int256 amount0, int256 amount1)
    {
        uint256 amt = uint256(amountSpecified);
        // Pay out 1:1, scaled by the fill, from whatever reward this pool holds.
        uint256 out = (amt * (askBps > 10_000 ? 10_000 : askBps)) / 10_000;
        IERC20Min(zeroForOne ? token1 : token0).transfer(recipient, out);
        int256 owe = int256((amt * askBps) / 10_000);
        (amount0, amount1) = zeroForOne ? (owe, int256(0)) : (int256(0), owe);
        for (uint256 i = 0; i < calls; i++) {
            ICallback(msg.sender).uniswapV3SwapCallback(amount0, amount1, data);
        }
    }

    /// The pool, calling anything it likes, outside any swap.
    function poke(address target, bytes calldata data) external returns (bool ok, bytes memory ret) {
        (ok, ret) = target.call(data);
    }
}

/// A claimant contract that makes a claim inside try/catch, so a reverted
/// claim does not revert the transaction around it -- the one arrangement
/// under which a leaked `_inFlight` could survive.
contract TryClaimer {
    function tryClaimV3(address dist, uint256 id, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (bool ok)
    {
        (ok,) = dist.call(abi.encodeWithSignature(
            "claimOnV3(uint256,uint256,bytes32[],uint256)", id, amount, proof, minOut));
    }
}

/// A claimant contract that borrows the gate for the length of one call and
/// hands it straight back, which is all a "holding" check at claim time asks.
contract GateBorrower {
    function borrowAndClaim(address token, address lender, uint256 gate, address dist, uint256 id,
                            uint256 amount, bytes32[] calldata proof) external {
        IERC20Min(token).transferFrom(lender, address(this), gate);
        (bool ok, bytes memory ret) = dist.call(abi.encodeWithSignature(
            "claim(uint256,uint256,bytes32[])", id, amount, proof));
        if (!ok) {
            assembly { revert(add(ret, 32), mload(ret)) }
        }
        IERC20Min(token).transfer(lender, gate);
    }
}
