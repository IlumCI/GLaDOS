// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/// @title Claim an epoch's allocation as a real market buy, and burn what arrives.
///
/// @notice For proving the loop works without anybody being enriched by it. A
/// share is mined, an epoch is funded, and the claim executes a genuine buy of
/// GLADOS on the real pool -- moving the real price, paying the real tax,
/// appearing on chain as a trade -- and then the tokens go to a burn address
/// instead of to a wallet. Nobody owns the result, including the operator.
///
/// @dev **Why this is a separate contract rather than a mode in the distributor.**
///
/// `GladosDistributor._admit` requires `msg.sender` to be the address in the
/// leaf, and `claimOnMarket` sends the bought token to `msg.sender`. So paying a
/// burn address directly is impossible in the obvious way: a burn address has no
/// private key and cannot call anything.
///
/// The distributor could have grown a recipient argument. It was not, for one
/// reason: it was audited and fixed three commits ago and has still never been
/// deployed, so the cheapest possible change to it is none. A leaf is an address
/// and what that address does with its allocation is the address's own business
/// -- which is exactly the seam this contract sits in. The distributor keeps the
/// shape somebody read.
///
/// **So this contract is the claimant.** It goes in the Merkle tree as a leaf
/// like any miner, and `claimAndBurn` is the thing anybody may call to make that
/// leaf collect and destroy its own reward.
///
/// **It is permissionless on purpose.** Every path out of here ends with tokens
/// at a burn address, so there is nothing for a caller to steal and no reason to
/// ask who they are -- whoever pays the gas triggers a burn. An `onlyOwner` here
/// would be a key to protect for no benefit, and the one thing that could go
/// wrong with it is losing that key and stranding the allocation until the
/// epoch's deadline.
interface IERC20 {
    function balanceOf(address) external view returns (uint256);
}

interface IGladosDistributor {
    function claimOnMarket(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256);
    function claimOnV3(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256);
}

contract GladosBurner {
    /// The distributor whose epochs this claims from.
    address public immutable distributor;
    /// The token that arrives and is destroyed. Usually GLADOS; on a `MarketV3`
    /// epoch it is whatever the epoch names as its reward.
    address public immutable reward;

    /// Where it goes.
    ///
    /// **`0xdEaD` and not the zero address**, and the difference is not
    /// cosmetic: a great many ERC-20s -- every OpenZeppelin one, and launchpad
    /// tokens generally -- refuse a transfer to `address(0)` outright. A burner
    /// pointed there reverts on the last line after the swap has already moved
    /// the price, which is the worst place in this sequence to fail. `0xdEaD` is
    /// an ordinary address with no known key, so a transfer to it is an ordinary
    /// transfer and every token allows it.
    address public constant BURN = 0x000000000000000000000000000000000000dEaD;

    event Burnt(uint256 indexed epoch, uint256 bought, uint256 burnt);

    error NothingArrived();
    error BurnFailed();

    constructor(address distributor_, address reward_) {
        require(distributor_ != address(0) && reward_ != address(0), "zero address");
        distributor = distributor_;
        reward = reward_;
    }

    /// Claim from a `Market` epoch and burn the proceeds.
    ///
    /// `minOut` is passed through and must be non-zero -- the distributor refuses
    /// zero, deliberately, because a swap with no bound on a thin pool is a
    /// sandwich waiting to happen.
    function claimAndBurn(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256 burnt)
    {
        IGladosDistributor(distributor).claimOnMarket(epochId, amount, proof, minOut);
        return _burnEverything(epochId);
    }

    /// The same, for a `MarketV3` epoch paying a token the epoch names.
    function claimAndBurnV3(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256 burnt)
    {
        IGladosDistributor(distributor).claimOnV3(epochId, amount, proof, minOut);
        return _burnEverything(epochId);
    }

    /// Send the whole balance to the burn address.
    ///
    /// **The balance, not the amount the distributor reported.** Two reasons and
    /// both are measured facts about this token rather than caution: GLADOS taxes
    /// transfers, so what arrived is smaller than what the pool sent; and
    /// anything already sitting here from a previous call that somehow did not
    /// complete should go too, rather than accumulating in a contract nobody
    /// watches.
    function _burnEverything(uint256 epochId) private returns (uint256) {
        uint256 bal = IERC20(reward).balanceOf(address(this));
        if (bal == 0) revert NothingArrived();

        // `transfer` through a low-level call, for the reason
        // `GladosDistributor._move` gives: the standard says it returns `bool`
        // and enough tokens return nothing that a strict decode reverts on
        // transfers that succeeded. Empty return data is success; a short one is
        // a failure rather than a decode panic.
        (bool okCall, bytes memory ret) =
            reward.call(abi.encodeWithSelector(0xa9059cbb, BURN, bal));
        if (!okCall) revert BurnFailed();
        if (ret.length != 0 && (ret.length < 32 || !abi.decode(ret, (bool)))) revert BurnFailed();

        // Measured after the fact, because the tax applies here too and the
        // event should say what left rather than what was asked for -- which is
        // `design/audit.md`'s finding 4, not repeated.
        uint256 left = IERC20(reward).balanceOf(address(this));
        uint256 burnt = bal - left;
        emit Burnt(epochId, bal, burnt);
        return burnt;
    }

    /// What this holds right now, which should always be zero between calls.
    ///
    /// Exposed so an observer can check that rather than take it on trust: a
    /// non-zero balance here means a burn did not complete, and the next
    /// `claimAndBurn` will sweep it.
    function stranded() external view returns (uint256) {
        return IERC20(reward).balanceOf(address(this));
    }
}
