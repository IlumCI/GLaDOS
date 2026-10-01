// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Push an epoch's payouts to every miner in one transaction.
///
/// **This is the "delivered to your door" payout.** The distributor is a pull
/// contract: every miner claims, and pays gas to do it, which makes a small
/// equal share not worth collecting. Here the pool pays once, for everybody:
/// one market buy of the token, then one call to `send` with the whole list.
///
/// **It holds nothing, ever.** Tokens move straight from the caller to each
/// recipient with `transferFrom`; this contract is never a balance-holder, so
/// there is nothing in it to be stuck, drained or reclaimed, no owner, and no
/// state. The caller approves exactly the epoch's total and calls once.
///
/// **All or nothing.** One transfer that fails reverts the batch, so a payout
/// is either delivered to everyone on the list or to nobody -- never a quiet
/// partial that looks like success on a block explorer. The pool rebuilds the
/// list without whoever made it fail and tries again.
///
/// **Every recipient must receive exactly their amount.** `design/audit-2.md`
/// F4 is why: assuming a transfer moves exactly `amount` is true of GLADOS
/// wallet-to-wallet today and is a property of somebody else's contract. The
/// recipient's balance is measured either side of each transfer, so a token
/// that taxes or skims one of these reverts the batch instead of shorting a
/// miner by an amount nobody is told about.
contract GladosBatch {
    error LengthMismatch();
    error TransferFailed(uint256 index);
    error ShortPaid(uint256 index, uint256 wanted, uint256 got);

    event Sent(address indexed token, address indexed from, uint256 recipients, uint256 total);

    function send(address token, address[] calldata to, uint256[] calldata amounts) external {
        if (to.length != amounts.length) revert LengthMismatch();
        uint256 total;
        for (uint256 i = 0; i < to.length; ++i) {
            uint256 before = _balanceOf(token, to[i]);
            (bool ok, bytes memory ret) =
                token.call(abi.encodeWithSelector(0x23b872dd, msg.sender, to[i], amounts[i]));
            // Empty return data is success -- enough tokens return nothing that
            // a strict decode reverts on transfers that worked -- and a returned
            // word must be exactly 1 (the distributor's F7 rule).
            if (!ok || (ret.length != 0 && (ret.length < 32 || abi.decode(ret, (uint256)) != 1))) {
                revert TransferFailed(i);
            }
            uint256 got = _balanceOf(token, to[i]) - before;
            if (got != amounts[i]) revert ShortPaid(i, amounts[i], got);
            total += amounts[i];
        }
        emit Sent(token, msg.sender, to.length, total);
    }

    function _balanceOf(address token, address who) private view returns (uint256) {
        (bool ok, bytes memory ret) = token.staticcall(abi.encodeWithSelector(0x70a08231, who));
        if (!ok || ret.length < 32) revert TransferFailed(type(uint256).max);
        return abi.decode(ret, (uint256));
    }
}
