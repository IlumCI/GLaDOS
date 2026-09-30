// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title GladosReader
/// @notice Never deployed. Its creation code is sent as an `eth_call` with no
/// `to`: the constructor runs, reads every address it was given, and returns
/// the readings as if they were runtime code. One request answers what used
/// to take two JSON-RPC calls per address.
///
/// **Why it exists: the public 4663 RPC refused a 100-call batch** (HTTP 429,
/// measured), and the treasury was sending exactly those -- every balance read
/// would have been rate-limited, every miner's work carried forever, and
/// nobody paid. This reads 200 addresses in one call.
///
/// Per address, three words: the token balance (all ones if `balanceOf`
/// failed, which the reader treats as unreadable), the code size, and the
/// first 32 bytes of code -- enough to tell an EIP-7702 delegation
/// (`0xef0100` + an address, 23 bytes) from a contract.
contract GladosReader {
    constructor(address token, address[] memory who) {
        uint256 n = who.length;
        bytes memory out = new bytes(n * 96);
        for (uint256 i = 0; i < n; ++i) {
            address a = who[i];
            (bool ok, bytes memory r) = token.staticcall(abi.encodeWithSelector(0x70a08231, a));
            uint256 bal = ok && r.length >= 32 ? abi.decode(r, (uint256)) : type(uint256).max;
            uint256 size;
            bytes32 head;
            assembly {
                size := extcodesize(a)
                let p := mload(0x40)
                mstore(p, 0)
                extcodecopy(a, p, 0, 32)
                head := mload(p)
                let o := add(add(out, 32), mul(i, 96))
                mstore(o, bal)
                mstore(add(o, 32), size)
                mstore(add(o, 64), head)
            }
        }
        assembly {
            return(add(out, 32), mload(out))
        }
    }
}
