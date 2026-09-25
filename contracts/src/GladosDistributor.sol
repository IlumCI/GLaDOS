// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/// @title The claim contract for the GLaDOS mining pool's GLADOS reward.
///
/// @notice Layer 1 of this pool never holds a miner's coins: they are paid by
/// the upstream chain, directly, in what they mined. This contract is Layer 2,
/// and it distributes the *operator's own* fee revenue, converted to GLADOS.
/// Converting your own money is not a regulated service; holding somebody
/// else's is, which is why these are two layers and not one.
///
/// @dev The design is a Merkle distributor per epoch. What it does differently
/// from the usual one, and why:
///
/// **The gate is per epoch and immutable once the epoch exists.** A single
/// mutable `minimumBalance` would let the operator change the rules for work
/// already done. Here the threshold is fixed at the moment the root is
/// published, so an epoch's terms are as final as its amounts, and changing
/// the gate is visible as a new epoch rather than as a silent state write.
///
/// **The balance check happens at claim time, not at share time.** That is the
/// point of putting it here at all: `docs/token/index.html` argues that a
/// balance check compiled into a ring-0 kernel is one edit away from being
/// deleted, and that a check in the claim contract is not. It also means
/// selling below the gate between mining and claiming forfeits the claim, which
/// is intended -- the gate is a holding requirement, not an entry fee.
///
/// **Amounts are taken as received rather than as requested.** GLADOS is a tax
/// token. Wallet-to-wallet transfers are untaxed -- verified on-chain, by
/// reading `Transfer` logs and finding plain transfers that move an identical
/// amount with no second event, where a buy from the pair emits a 1% tax leg
/// and a sell 3% -- but a contract that *assumes* that and is wrong hands out
/// claims it cannot pay, and the failure lands on whoever claims last. So the
/// balance is measured either side of the transfer and the epoch records what
/// actually arrived.
/// The only ERC-20 call this contract makes through a typed interface.
///
/// `balanceOf` is a view and its return is unambiguous, so it can be typed.
/// The two that move tokens go through `_call` instead, because the standard
/// says they return `bool` and enough tokens return nothing that a strict
/// decode reverts on transfers that actually succeeded.
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
}

/// The Uniswap V2 pair itself, which is what this talks to instead of a router.
///
/// **There is no canonical V2 router on this chain**, checked rather than
/// assumed: the two contracts real buyers actually call, found by reading the
/// `to` of live swap transactions, answer neither `factory()` nor `WETH()`, so
/// they are aggregators or the launchpad's own periphery rather than the
/// router this would otherwise need.
///
/// Going direct is the better answer anyway and would have been even with a
/// router available. A router is a convenience wrapper that computes the output
/// and moves the input; doing both here removes a third-party contract from the
/// path, removes the token approval that contract would need, and removes the
/// question of which router is the real one -- which on a young chain is a
/// question with an expensive wrong answer.
interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
    // Asked once, in the constructor, and never again. A pair cannot be
    // *derived* from two token addresses without trusting a factory, but it can
    // be *checked* against them -- which is a different and much cheaper claim,
    // and the one the constructor was missing.
    function token0() external view returns (address);
    function token1() external view returns (address);
    function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data) external;
}

/// Uniswap V3's pool, which pays out first and asks to be paid back.
///
/// That inversion is the whole reason this needed a second code path rather
/// than a second address. A V2 pair is sent the input and then told to send the
/// output; a V3 pool sends the output and then calls `uniswapV3SwapCallback` on
/// whoever asked, which must move the input before that call returns or the
/// entire swap unwinds. So a contract trading on V3 has to expose a function
/// whose job is to pay somebody, and the authorisation on that function is the
/// only thing standing between this contract's balance and anyone who asks for
/// it. See `_inFlight`.
interface IUniswapV3Pool {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function fee() external view returns (uint24);
    function swap(
        address recipient,
        bool zeroForOne,
        int256 amountSpecified,
        uint160 sqrtPriceLimitX96,
        bytes calldata data
    ) external returns (int256 amount0, int256 amount1);
}

/// Only `getPool`, because that is the one question worth asking it: does this
/// factory consider this address to be the pool for these three keys.
interface IUniswapV3Factory {
    function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address);
}

contract GladosDistributor {
    /// How an epoch pays.
    ///
    /// **`Market` is not a better `Direct`, it is a different trade**, and the
    /// arithmetic decides rather than taste. A swap costs about $0.048 of gas
    /// on this chain against $0.029 for a transfer, and a claim is worth what
    /// the fee pot divided by the miners says it is worth: $0.0021 each over a
    /// 36-hour event, where the gas is 2,286% of the reward, and $0.51 each on
    /// a yearly epoch with a thousand miners, where it is 9%.
    ///
    /// What `Market` buys is that every claim is a real trade -- it moves the
    /// price, it pays the token's own buy tax into dividends, liquidity and the
    /// burn, and it is visible on-chain as a buy rather than as an operator
    /// handing out tokens they converted somewhere nobody watched. What it
    /// costs is gas per claim and a price that moves between the first
    /// claimant and the last.
    enum Mode {
        /// The epoch holds the reward token. A claim transfers it.
        Direct,
        /// The epoch holds an input token. A claim swaps it on the market and
        /// the reward token goes straight to the claimant.
        Market,
        /// As `Market`, on a Uniswap V3 pool, and paying a reward token the
        /// epoch names rather than the one this contract gates on.
        ///
        /// **Those two changes travel together on purpose.** V3 is where the
        /// tokenized-equity pools live on 4663 -- the V2 market for them is
        /// four hundredths of a percent of the real one -- and the reason to
        /// reach them at all is to pay somebody in NVDA while still gating on
        /// GLADOS. A mode that swapped venue without splitting reward from gate
        /// would have no caller.
        MarketV3
    }

    struct Epoch {
        /// Merkle root over `(account, amount)` leaves.
        bytes32 root;
        /// What the contract actually received when the epoch was opened.
        uint256 funded;
        /// Sum of what has been claimed so far.
        uint256 claimed;
        /// Minimum GLADOS balance a claimant must hold, checked at claim time.
        uint256 gate;
        /// After this, the operator may take back what nobody claimed.
        uint64 deadline;
        /// Whether the remainder has already gone back.
        bool reclaimed;
        /// How this epoch pays. Fixed when the epoch opens, like the gate.
        Mode mode;
        /// `MarketV3` only: the pool a claim swaps through, checked against the
        /// factory when the epoch opened. Zero in every other mode.
        address pool;
        /// `MarketV3` only: what a claim buys. Zero in every other mode, where
        /// the reward is `token` and there is nothing to choose.
        address reward;
    }

    /// The token being distributed. Immutable: a distributor that could be
    /// repointed at a different token is a different contract wearing this
    /// one's address.
    address public immutable token;

    /// Who may open epochs and reclaim expired remainders. Immutable for the
    /// same reason -- there is no upgrade path and no admin key rotation, so
    /// the whole of what this address can do is visible in two functions.
    address public immutable operator;

    /// The pool. Immutable: a distributor that could be repointed at a
    /// different pair is one that could route a claim into a pool the operator
    /// controls. Zero disables `Market` epochs entirely, which is the right
    /// configuration on a chain with no pool worth using.
    address public immutable pair;

    /// Whether `quote` is the pair's `token0`, decided once at construction.
    ///
    /// A V2 pair orders its two tokens by address and `swap` takes the outputs
    /// positionally, so getting this backwards asks the pool to pay out the
    /// token being paid *in*. Computed from the addresses rather than read from
    /// the pair, because the ordering rule is the pair's own and a constructor
    /// that trusted a call here would trust the thing it is checking.
    bool public immutable quoteIsToken0;

    /// What a `Market` epoch is funded in. WETH here, because that is what the
    /// GLADOS pool is quoted against -- read from the token's own
    /// `quoteToken()` rather than chosen, so the path is the pool that exists.
    address public immutable quote;

    /// The V3 factory, and the only reason a pool address is worth believing.
    ///
    /// Immutable for the reason `pair` is: a distributor that could be
    /// repointed at a different factory could be pointed at one whose `getPool`
    /// says yes to anything. Zero disables `MarketV3` entirely, which is the
    /// right configuration on a chain that has no V3 deployment.
    address public immutable v3Factory;

    /// The pool a swap is in flight with, and the whole of the callback's
    /// authorisation.
    ///
    /// Set immediately before `swap` and cleared immediately after, so it holds
    /// a non-zero value only inside one call this contract itself started. A
    /// callback arriving at any other moment finds zero and reverts.
    ///
    /// **Authorising by factory lookup instead would be the bug**, and it is
    /// the standard way this callback gets drained: `getPool` says yes to every
    /// genuine pool, including one an attacker deployed for a pair of worthless
    /// tokens of their own, from which they can call this callback and be paid
    /// in `quote` for nothing.
    address private _inFlight;

    /// The two ends of V3's price range, from the reference `TickMath`.
    ///
    /// A `swap` must name a price limit, and these are the values that mean "no
    /// limit" in each direction. Passing them is deliberate: the protection a
    /// claimant actually gets is `minOut`, measured on their own balance after
    /// the fact, and a second bound expressed as a square-rooted Q64.96 price
    /// would be a number no caller could sanity-check.
    uint160 private constant MIN_SQRT_RATIO = 4295128739;
    uint160 private constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;

    Epoch[] private _epochs;

    /// `epoch => account => claimed`. By address rather than by a bitmap over
    /// leaf indices: the index form saves gas and buys a way for two leaves to
    /// name one account, and on a chain where a claim costs about three cents
    /// that is the wrong side of the trade.
    mapping(uint256 => mapping(address => bool)) public hasClaimed;

    event EpochOpened(uint256 indexed epoch, bytes32 root, uint256 funded, uint256 gate, uint64 deadline, Mode mode);
    /// `amount` is the leaf and `received` is what reached the claimant. Equal
    /// under `Direct`; under `Market` the leaf is denominated in `quote`, so
    /// the two together are the record of what the market actually gave --
    /// which is the whole point of paying through one.
    event Claimed(uint256 indexed epoch, address indexed account, uint256 amount, uint256 received);
    event Reclaimed(uint256 indexed epoch, uint256 amount);

    error NotOperator();
    error NoRoot();
    error NothingFunded();
    error DeadlineInPast();
    error NoSuchEpoch();
    error AlreadyClaimed();
    error BadProof();
    error BelowGate(uint256 held, uint256 needed);
    error EpochClosed();
    error EpochOpen();
    error AlreadyReclaimed();
    error Insolvent(uint256 want, uint256 have);
    error TransferFailed();
    error NoMarket();
    error WrongMode();
    error NoSlippageBound();
    error TooLittleOut(uint256 got, uint256 wanted);
    error BadPool();
    error BadCallback();

    constructor(address token_, address operator_, address pair_, address quote_, address v3Factory_) {
        require(token_ != address(0) && operator_ != address(0), "zero address");
        // `pair` and `quote` may both be zero, which simply means this
        // distributor cannot open `Market` epochs. Requiring them would make
        // the contract undeployable on a chain where the pool does not exist
        // yet, which is a state this project has been in twice.
        require((pair_ == address(0)) == (quote_ == address(0)), "pair and quote go together");
        // **The pair is checked against the two tokens it is supposed to hold.**
        // `design/audit.md` finding 1: without this, `quoteIsToken0` below is a
        // comparison of two addresses that is never reconciled with what the
        // pair actually contains, and a wrong `pair` produces a distributor
        // which accepts funding for market epochs and refuses every claim
        // forever -- driven, `TooLittleOut(0,1)`. `pair` is immutable, so the
        // constructor is the only place this can ever be caught.
        //
        // `openEpochOnV3` has checked its pool this way since it was written.
        // The asymmetry was the finding.
        if (pair_ != address(0)) {
            address p0 = IUniswapV2Pair(pair_).token0();
            address p1 = IUniswapV2Pair(pair_).token1();
            require(
                (p0 == quote_ && p1 == token_) || (p0 == token_ && p1 == quote_),
                "pair does not hold quote and token"
            );
        }
        token = token_;
        operator = operator_;
        pair = pair_;
        quote = quote_;
        quoteIsToken0 = quote_ < token_;
        // Deliberately not tied to `quote`: a chain can have a V3 deployment
        // and no V2 pair for this token, or the reverse, and refusing either
        // combination would make the contract undeployable for a reason that
        // has nothing to do with the epoch being opened.
        v3Factory = v3Factory_;
    }

    modifier onlyOperator() {
        if (msg.sender != operator) revert NotOperator();
        _;
    }

    /// Publish a root and fund it in one transaction.
    ///
    /// One call rather than "create then fund", because an epoch that exists
    /// and is not funded is a set of claims that revert, and the person who
    /// finds out is a miner rather than the operator.
    ///
    /// The operator must have approved this contract for `amount` first.
    function openEpoch(bytes32 root, uint256 amount, uint256 gate, uint64 deadline)
        external
        onlyOperator
        returns (uint256 epochId)
    {
        if (root == bytes32(0)) revert NoRoot();
        if (amount == 0) revert NothingFunded();
        if (deadline <= block.timestamp) revert DeadlineInPast();

        // Measured rather than assumed. See the note on tax tokens above.
        uint256 before = IERC20(token).balanceOf(address(this));
        _pull(msg.sender, amount);
        uint256 got = IERC20(token).balanceOf(address(this)) - before;
        if (got == 0) revert NothingFunded();

        epochId = _epochs.length;
        _epochs.push(Epoch({
            root: root,
            funded: got,
            claimed: 0,
            gate: gate,
            deadline: deadline,
            reclaimed: false,
            mode: Mode.Direct,
            pool: address(0),
            reward: address(0)
        }));
        emit EpochOpened(epochId, root, got, gate, deadline, Mode.Direct);
    }

    /// Open an epoch that pays by buying on the market, one buy per claim.
    ///
    /// Funded in `quote` rather than in the reward token: the operator never
    /// converts anything, so there is no conversion rate for anybody to have to
    /// trust. Each claimant's own transaction is the trade, at the price the
    /// pool has at that moment, and the reward token goes from the pair to them
    /// -- which means the token's buy tax applies and feeds whatever the token
    /// does with it.
    ///
    /// **The leaf is denominated in `quote`, and that has a consequence worth
    /// stating before somebody discovers it.** Every claim moves the price, so
    /// the first claimant gets more reward token per unit of quote than the
    /// last. That is a race, it is inherent to paying through a market rather
    /// than around one, and it is the reason `Direct` still exists.
    function openEpochOnMarket(bytes32 root, uint256 amount, uint256 gate, uint64 deadline)
        external
        onlyOperator
        returns (uint256 epochId)
    {
        if (pair == address(0)) revert NoMarket();
        if (root == bytes32(0)) revert NoRoot();
        if (amount == 0) revert NothingFunded();
        if (deadline <= block.timestamp) revert DeadlineInPast();

        uint256 before = IERC20(quote).balanceOf(address(this));
        _move(quote, abi.encodeWithSelector(0x23b872dd, msg.sender, address(this), amount));
        uint256 got = IERC20(quote).balanceOf(address(this)) - before;
        if (got == 0) revert NothingFunded();

        epochId = _epochs.length;
        _epochs.push(Epoch({
            root: root,
            funded: got,
            claimed: 0,
            gate: gate,
            deadline: deadline,
            reclaimed: false,
            mode: Mode.Market,
            pool: address(0),
            reward: address(0)
        }));
        emit EpochOpened(epochId, root, got, gate, deadline, Mode.Market);
    }

    /// Open an epoch that pays a token this contract does not gate on, bought
    /// on a Uniswap V3 pool.
    ///
    /// Funded in `quote`, like `openEpochOnMarket`, and for the same reason:
    /// the operator never holds the reward and every claimant's purchase is a
    /// buy on the open market that anybody can see.
    ///
    /// **The pool is verified rather than trusted, and the check is worth
    /// reading.** Anything can be deployed at an address and answer `swap`.
    /// Reading the candidate's own three keys and asking the factory to name
    /// the pool for those keys is a check a liar cannot pass, because a lie
    /// about any key resolves to a different address than the one being asked
    /// about. Then the keys must be the pair this epoch claims, or an operator
    /// could point a NVDA epoch at a real pool for something else entirely.
    function openEpochOnV3(
        bytes32 root,
        uint256 amount,
        uint256 gate,
        uint64 deadline,
        address pool_,
        address reward_
    ) external onlyOperator returns (uint256 epochId) {
        if (v3Factory == address(0) || quote == address(0)) revert NoMarket();
        if (root == bytes32(0)) revert NoRoot();
        if (amount == 0) revert NothingFunded();
        if (deadline <= block.timestamp) revert DeadlineInPast();
        if (reward_ == address(0) || reward_ == quote) revert BadPool();

        address t0 = IUniswapV3Pool(pool_).token0();
        address t1 = IUniswapV3Pool(pool_).token1();
        if (IUniswapV3Factory(v3Factory).getPool(t0, t1, IUniswapV3Pool(pool_).fee()) != pool_) revert BadPool();
        if (!((t0 == quote && t1 == reward_) || (t0 == reward_ && t1 == quote))) revert BadPool();

        uint256 before = IERC20(quote).balanceOf(address(this));
        _move(quote, abi.encodeWithSelector(0x23b872dd, msg.sender, address(this), amount));
        uint256 got = IERC20(quote).balanceOf(address(this)) - before;
        if (got == 0) revert NothingFunded();

        epochId = _epochs.length;
        _epochs.push(Epoch({
            root: root,
            funded: got,
            claimed: 0,
            gate: gate,
            deadline: deadline,
            reclaimed: false,
            mode: Mode.MarketV3,
            pool: pool_,
            reward: reward_
        }));
        emit EpochOpened(epochId, root, got, gate, deadline, Mode.MarketV3);
    }

    /// Claim from a `MarketV3` epoch, swapping the allocation on the way out.
    ///
    /// `minOut` carries the same rule as `claimOnMarket` and is measured the
    /// same way, on the claimant's own balance after the fact. That matters
    /// more here than it did there: the reward may be a token with a transfer
    /// hook this contract has never seen, since the operator names it per
    /// epoch, so what the pool reports having sent is not evidence of what
    /// arrived.
    function claimOnV3(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256 received)
    {
        if (minOut == 0) revert NoSlippageBound();
        Epoch storage e = _admit(epochId, amount, proof);
        if (e.mode != Mode.MarketV3) revert WrongMode();

        address pool_ = e.pool;
        address reward_ = e.reward;
        // Recomputed rather than stored: the ordering is a property of the two
        // addresses and storing it would be a second place for it to be wrong.
        bool zeroForOne = quote < reward_;

        uint256 before = IERC20(reward_).balanceOf(msg.sender);
        // **Saved and restored rather than set and cleared**, which is
        // `design/audit.md` finding 6. A V3 pool pays the recipient *before* it
        // calls back, so a reward token with a transfer hook runs code inside
        // this swap; if that code enters here again and succeeds, the inner call
        // used to clear this to zero on its way out and the outer pool's
        // callback then found nothing and was refused. Funds never moved -- the
        // whole transaction unwound -- but an epoch whose reward token has a
        // hook was unclaimable, and the fix is this one line.
        address prev = _inFlight;
        _inFlight = pool_;
        IUniswapV3Pool(pool_).swap(
            msg.sender,
            zeroForOne,
            int256(amount),
            zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1,
            ""
        );
        // Restored on the way out rather than zeroed: a stale value here is a
        // standing authorisation to be paid, and an unconditional zero is what
        // broke a nested claim.
        _inFlight = prev;

        received = IERC20(reward_).balanceOf(msg.sender) - before;
        if (received < minOut) revert TooLittleOut(received, minOut);
        emit Claimed(epochId, msg.sender, amount, received);
    }

    /// Pay the pool for a swap this contract is in the middle of.
    ///
    /// The deltas are signed from the pool's point of view, so the positive one
    /// is what the pool is owed. Only one of the two can be positive on an
    /// exact-input swap, and it is always `quote` here, because that is the
    /// only thing any epoch pays in.
    ///
    /// Everything protecting this is the `_inFlight` check on the first line.
    function uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes calldata) external {
        if (msg.sender != _inFlight || _inFlight == address(0)) revert BadCallback();
        int256 owed = amount0Delta > 0 ? amount0Delta : amount1Delta;
        // A pool asking for nothing, or paying us on both legs, is not a swap
        // this contract started correctly. Refuse rather than send zero and
        // let the pool decide what that meant.
        if (owed <= 0) revert BadCallback();
        _send(quote, msg.sender, uint256(owed));
    }

    /// Claim one epoch's allocation.
    ///
    /// `amount` and `proof` come from the published root; anybody can compute
    /// them from the share log, which is the point of publishing it.
    function claim(uint256 epochId, uint256 amount, bytes32[] calldata proof) external {
        Epoch storage e = _admit(epochId, amount, proof);
        if (e.mode != Mode.Direct) revert WrongMode();
        // **Measured, not assumed**, which is `design/audit.md` finding 4. This
        // header argues at length that amounts must be taken as received because
        // GLADOS is a tax token, and applies it on every inbound leg and on both
        // market claims -- and this was the one path that emitted the number it
        // asked for. A `Claimed` event is what an indexer, a miner and any
        // published accounting read, so `received` is the field that exists to
        // be true.
        uint256 before = IERC20(token).balanceOf(msg.sender);
        _send(token, msg.sender, amount);
        uint256 got = IERC20(token).balanceOf(msg.sender) - before;
        emit Claimed(epochId, msg.sender, amount, got);
    }

    /// Claim by buying on the market, in the claimant's own transaction.
    ///
    /// `minOut` is the claimant's slippage bound and must be non-zero. A swap
    /// with no bound on a pool this thin is a sandwich waiting to happen, and
    /// zero is never the right answer -- refusing it costs a revert and saves
    /// somebody their reward.
    function claimOnMarket(uint256 epochId, uint256 amount, bytes32[] calldata proof, uint256 minOut)
        external
        returns (uint256 received)
    {
        if (minOut == 0) revert NoSlippageBound();
        Epoch storage e = _admit(epochId, amount, proof);
        if (e.mode != Mode.Market) revert WrongMode();

        // The whole swap, without a router: send the input to the pair, work
        // out what it owes from its own reserves, and ask for it.
        (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
        (uint256 rIn, uint256 rOut) = quoteIsToken0 ? (uint256(r0), uint256(r1)) : (uint256(r1), uint256(r0));
        uint256 out = _amountOut(amount, rIn, rOut);
        if (out == 0) revert TooLittleOut(0, minOut);

        _send(quote, pair, amount);
        uint256 before = IERC20(token).balanceOf(msg.sender);
        // `swap` takes outputs positionally in the pair's own token order, and
        // asking for the wrong one asks the pool to pay out what is being paid
        // in. Hence `quoteIsToken0`, fixed at construction.
        if (quoteIsToken0) {
            IUniswapV2Pair(pair).swap(0, out, msg.sender, "");
        } else {
            IUniswapV2Pair(pair).swap(out, 0, msg.sender, "");
        }

        // **Measured on the claimant, not taken from `out`.** The pair sends
        // `out` and the token takes its buy tax out of that transfer, so what
        // arrives is smaller -- and by an amount this contract has no business
        // predicting, since the tax rate is the token's and can be whatever it
        // is on the day.
        received = IERC20(token).balanceOf(msg.sender) - before;
        if (received < minOut) revert TooLittleOut(received, minOut);
        emit Claimed(epochId, msg.sender, amount, received);
    }

    /// Everything both claim paths must do, in one place.
    ///
    /// Written once because two copies of a gate check is how one of them ends
    /// up a `>` where the other is a `>=`. It marks the claim *before*
    /// returning, so both callers are reentrancy-safe by the time they touch a
    /// token.
    function _admit(uint256 epochId, uint256 amount, bytes32[] calldata proof)
        private
        returns (Epoch storage e)
    {
        if (epochId >= _epochs.length) revert NoSuchEpoch();
        e = _epochs[epochId];
        if (block.timestamp > e.deadline) revert EpochClosed();
        if (hasClaimed[epochId][msg.sender]) revert AlreadyClaimed();

        // **The gate, and the whole reason this lives on-chain.**
        uint256 held = IERC20(token).balanceOf(msg.sender);
        if (held < e.gate) revert BelowGate(held, e.gate);

        if (!_verify(proof, e.root, _leaf(msg.sender, amount))) revert BadProof();

        // A root that promises more than the epoch holds is an operator error,
        // caught here rather than by the last claimant getting a failed
        // transfer they cannot explain.
        uint256 left = e.funded - e.claimed;
        if (amount > left) revert Insolvent(amount, left);

        // Effects before interaction. GLADOS is a tax token and tax tokens run
        // code on transfer, so the reentrant path is real rather than
        // theoretical -- and with this ordering a reentrant call finds
        // `hasClaimed` already true.
        hasClaimed[epochId][msg.sender] = true;
        e.claimed += amount;
    }

    /// Take back what nobody claimed, once the epoch has closed.
    ///
    /// **This is the one power the operator has over funds already committed,
    /// and it is bounded in three ways**: only after a deadline fixed when the
    /// epoch was opened, only the unclaimed remainder of that epoch, and only
    /// once. Without it, tokens allocated to an address that lost its key are
    /// removed from supply forever with nobody having decided that; with it
    /// unbounded, the epoch would be a promise the operator could withdraw.
    function reclaim(uint256 epochId) external onlyOperator returns (uint256 amount) {
        if (epochId >= _epochs.length) revert NoSuchEpoch();
        Epoch storage e = _epochs[epochId];
        if (block.timestamp <= e.deadline) revert EpochOpen();
        if (e.reclaimed) revert AlreadyReclaimed();

        amount = e.funded - e.claimed;
        e.reclaimed = true;
        // The epoch's own asset: a `Market` epoch holds `quote`, and sending it
        // back as `token` would be sending tokens it does not have.
        if (amount > 0) _send(e.mode == Mode.Direct ? token : quote, operator, amount);
        emit Reclaimed(epochId, amount);
    }

    // ---------------------------------------------------------------- views

    function epochCount() external view returns (uint256) {
        return _epochs.length;
    }

    function epochs(uint256 id) external view returns (Epoch memory) {
        if (id >= _epochs.length) revert NoSuchEpoch();
        return _epochs[id];
    }

    /// What a claim would do, without doing it. Every reason a claim can fail,
    /// answered as a string, because the thing a miner needs is not a revert
    /// selector -- it is to know whether the problem is their balance, their
    /// proof, or the clock.
    /// Answers `mode` as well, and that is `design/audit.md` finding 2.
    ///
    /// It used to say only yes or no, and it never read `mode` -- so it approved
    /// a market epoch's claim and `claim()` then reverted `WrongMode`. It cannot
    /// know which function a caller intends, so the honest fix is to say which
    /// one applies rather than to guess: `Direct` takes `claim`, `Market` takes
    /// `claimOnMarket`, `MarketV3` takes `claimOnV3`.
    function checkClaim(uint256 epochId, address account, uint256 amount, bytes32[] calldata proof)
        external
        view
        returns (bool ok, string memory reason, Mode mode)
    {
        if (epochId >= _epochs.length) return (false, "no such epoch", Mode.Direct);
        Epoch storage e = _epochs[epochId];
        if (block.timestamp > e.deadline) return (false, "epoch closed", e.mode);
        if (hasClaimed[epochId][account]) return (false, "already claimed", e.mode);
        if (IERC20(token).balanceOf(account) < e.gate) return (false, "below the gate", e.mode);
        if (!_verify(proof, e.root, _leaf(account, amount))) {
            return (false, "proof does not match the root", e.mode);
        }
        if (amount > e.funded - e.claimed) return (false, "epoch is short", e.mode);
        return (true, "", e.mode);
    }

    /// The leaf preimage, exposed so an off-chain builder can be checked
    /// against this contract rather than against its own idea of the format.
    function leafOf(address account, uint256 amount) external pure returns (bytes32) {
        return _leaf(account, amount);
    }

    /// Verification, exposed for the same reason and it is the more important
    /// of the two.
    ///
    /// A builder checked only against itself is checked against nothing: its
    /// proofs verify under its own rules whatever those rules are. The tree
    /// tests were doing exactly that until this existed -- folding a JS proof
    /// with JS hashing and comparing to a JS root, which would pass with the
    /// handedness reversed, the leaf hashed once, or the odd node duplicated.
    /// This makes the arbiter the bytecode that will actually hold the tokens.
    function verifyProof(bytes32[] calldata proof, bytes32 root, address account, uint256 amount)
        external
        pure
        returns (bool)
    {
        return _verify(proof, root, _leaf(account, amount));
    }

    // ------------------------------------------------------------ internals

    /// **Double-hashed**, which is not decoration.
    ///
    /// With sorted-pair hashing an internal node is `keccak256` of 64 bytes. A
    /// singly-hashed leaf over `abi.encode(address, uint256)` is also 64 bytes,
    /// so a crafted "leaf" could be presented as an internal node and a proof
    /// forged around it. Hashing twice makes a leaf preimage 32 bytes and the
    /// two domains cannot collide.
    function _leaf(address account, uint256 amount) private pure returns (bytes32) {
        return keccak256(bytes.concat(keccak256(abi.encode(account, amount))));
    }

    /// Sorted-pair Merkle verification. Sorted so a proof carries no direction
    /// bits and the builder cannot disagree with the verifier about handedness,
    /// which is the mistake that produces a root that is wrong only sometimes.
    function _verify(bytes32[] calldata proof, bytes32 root, bytes32 leaf) private pure returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            h = h <= p ? keccak256(abi.encode(h, p)) : keccak256(abi.encode(p, h));
        }
        return h == root;
    }

    function _pull(address from, uint256 amount) private {
        _move(token, abi.encodeWithSelector(0x23b872dd, from, address(this), amount));
    }

    function _send(address asset, address to, uint256 amount) private {
        _move(asset, abi.encodeWithSelector(0xa9059cbb, to, amount));
    }

    /// Uniswap V2's constant-product formula with its 0.3% fee, written out.
    ///
    /// Written out rather than fetched from the pair, because a pair does not
    /// offer it -- the arithmetic lives in the router, which is the thing this
    /// contract is doing without. Three multiplications and a division, and the
    /// numbers it works on came from `getReserves()` in the same transaction,
    /// so there is no stale-price window between reading and swapping.
    function _amountOut(uint256 amountIn, uint256 rIn, uint256 rOut) private pure returns (uint256) {
        if (amountIn == 0 || rIn == 0 || rOut == 0) return 0;
        uint256 withFee = amountIn * 997;
        return (withFee * rOut) / (rIn * 1000 + withFee);
    }

    /// A transfer that accepts both conventions.
    ///
    /// The ERC-20 standard says `transfer` returns `bool`; enough tokens return
    /// nothing that a strict decode reverts on transfers that succeeded. Empty
    /// return data is treated as success and any other return must decode to
    /// true, which is the ordinary safe-transfer rule written out rather than
    /// imported.
    function _move(address asset, bytes memory data) private {
        (bool okCall, bytes memory ret) = asset.call(data);
        if (!okCall) revert TransferFailed();
        // **The length is checked before the decode.** `abi.decode` of fewer
        // than 32 bytes reverts with a panic rather than `TransferFailed`, so a
        // token returning a short value produced the wrong error -- which costs
        // whoever is reading a failed claim at three in the morning. Empty is
        // still success, because that is the second ERC-20 convention and the
        // reason this helper exists at all.
        if (ret.length != 0 && (ret.length < 32 || !abi.decode(ret, (bool)))) {
            revert TransferFailed();
        }
    }
}
