// What a miner can choose to be paid in, and how each choice is bought.
//
// One table, read by the gate (which records a wallet's choice from its hello),
// the treasury (which buys and sends it) and the tests. The kernel miner carries
// the same codes in src/mine/reward.rs; rewards.test.mjs reads that file and
// fails if the two lists drift.
//
// **Every token here was checked on chain 4663 on 2026-10-01**, by enumerating
// all 438,340 Uniswap V3 pools, keeping the 5,855 tokens with a USDG pool,
// and then only:
//   - genuine Robinhood stock tokens: an EIP-1967 beacon proxy over the stock
//     token beacon 0xe10b6f6b..., the one thing a copycat cannot fake by name;
//   - the deepest USDG pool for each, by USDG held in the pool;
//   - a route ETH -> USDG -> token that Uniswap's QuoterV2 quotes a fill for.
// GladosRwaPayout re-checks every pool against the factory before swapping, so
// a stale address here fails safe: that group is paid in $GLADOS instead.

export const CHAIN = {
  weth: "0x0bd7d308f8e1639fab988df18a8011f41eacad73",
  usdg: "0x5fc5360d0400a0fd4f2af552add042d716f1d168",
  v3Factory: "0x1f7d7550b1b028f7571e69a784071f0205fd2efa",
  quoterV2: "0x33e885ed0ec9bf04ecfb19341582aadcb4c8a9e7",
  // Deepest WETH/USDG V3 pool (fee 100, ~$15M USDG side).
  wethUsdgPool: "0x52e65b17fb6e5ba00ed806f37afcd2daa50271ca",
  wethUsdgFee: 100,
  // Every stock token is a proxy over this beacon, and its blocklist lives on
  // it (design/rwa.md): a blocked wallet would make a payout to it revert.
  stockBeacon: "0xe10b6f6b275de231345c20d14ab812db62151b00",
};

// symbol -> the token and its deepest USDG pool.
export const TOKENS = {
  NVDA: { token: "0xd0601ce157db5bdc3162bbac2a2c8af5320d9eec", pool: "0xd4eb21209c4d6093f80b5b84f5c45cc093ea14a3", fee: 500 },
  SPCX: { token: "0x4a0e65a3eccec6dbe60ae065f2e7bb85fae35eea", pool: "0xc61284332117c3fb23a2a56cceffd07f7af60029", fee: 500 },
  GOOGL: { token: "0x2e0847e8910a9732eb3fb1bb4b70a580adad4fe3", pool: "0x34d0dc122cf9a8eb296fc5e0d3a233625d7d19b7", fee: 500 },
  AMZN: { token: "0x12f190a9f9d7d37a250758b26824b97ce941bf54", pool: "0x8ac92da74ab5f3b1d024dc1943ad7e15dc4179ef", fee: 3000 },
  GME: { token: "0x1b0e319c6a659f002271b69db8a7df2f911c153e", pool: "0xe9713f453adb9245b19559790c96f470a18f2fdf", fee: 10000 },
  SPY: { token: "0x117cc2133c37b721f49de2a7a74833232b3b4c0c", pool: "0xa7bb1ac63bbab0c44316e6c8c455213441689167", fee: 500 },
  AMD: { token: "0x86923f96303d656e4aa86d9d42d1e57ad2023fdc", pool: "0x48d284a2a4d3dc1b3da08231fe44317e7e7aa51f", fee: 3000 },
  INTC: { token: "0xc72b96e0e48ecd4dc75e1e45396e26300bc39681", pool: "0x2e5a92f5013a64661a49312111be2e8abd33f56a", fee: 3000 },
  MU: { token: "0xff080c8ce2e5feadaca0da81314ae59d232d4afd", pool: "0xd057b1bc54917855bbee58ead58647f47cab35e5", fee: 3000 },
  AVGO: { token: "0x156e175dd063a8ce274c50654ef40e0032b3fbcf", pool: "0x5b7c404f1d7d77f9f3885ab13d7764f8a173028c", fee: 3000 },
  TSM: { token: "0x58ffe4a942d3885baa22d7520691f611ef09e7aa", pool: "0x07e8ea83d4c1340774c8965125e26e12bf943bf1", fee: 10000 },
  ASML: { token: "0x47f93d52cbec7c6d2cfc080e154002370a60daea", pool: "0xedb22516b14eb2d1c86927db373b0e8bf70f5cd1", fee: 10000 },
  MRVL: { token: "0x62fd0668e10d8b72339be2dcf7643001688ff13b", pool: "0x06cc0b96be1fa1d754ce2e1228f5d8c616f795b0", fee: 10000 },
  SKHY: { token: "0x84cab63bc87912e71ad199ff14a0ba45de68fef8", pool: "0x5f2a5025feb93a4c44da4e3b37e4fd3fb0ab5171", fee: 3000 },
  DELL: { token: "0x941ae714ec6d8130c7b75d67160ca08f1e7d11dd", pool: "0xc30c89cb7815a1488b7998d15eec73961707fc5a", fee: 10000 },
  MSFT: { token: "0xe93237c50d904957cf27e7b1133b510c669c2e74", pool: "0xeb60bcd1d920ad6e102690ccfc6fb488899e1510", fee: 3000 },
  AAPL: { token: "0xaf3d76f1834a1d425780943c99ea8a608f8a93f9", pool: "0xaae0d815ee56e4092a5e5c2911e676fea50b2d6d", fee: 500 },
  IBM: { token: "0x980dcf6766fa79f5cf0c4aadb3ab477ff15a9619", pool: "0x8cd848ce18b829c5c769aff27164078bb52e0e97", fee: 3000 },
  BB: { token: "0x48e39e56acdba37b09020c0b734a613c9a2f100a", pool: "0xc424e56e816f148a8b30ccbde2128e04d21f51e9", fee: 10000 },
  QQQ: { token: "0xd5f3879160bc7c32ebb4dc785f8a4f505888de68", pool: "0xd60a5d14db690b7afad71f76b108071d7175597d", fee: 500 },
  GLD: { token: "0xc9a981fee1f9dec688bb123ccdecc63d0debfc4e", pool: "0x7a6a053eccf1446a2633e05aa6d40d09381997ec", fee: 3000 },
  SLV: { token: "0x411efb0e7f985935daec3d4c3ebaea0d0ad7d89f", pool: "0x8cb787e6c315d464775289bad00fdd67d53ecb3d", fee: 3000 },
  USO: { token: "0xa30fa36db767ad9ed3f7a60fc79526fb4d56d344", pool: "0x02175608f1b5e6b5ed221ccfdc7be197d111d915", fee: 3000 },
};

// code -> what the miner sees, and the tokens a payout is split across
// evenly. `glados` is the default and buys through GladosPayout as before.
export const MENU = {
  glados: { name: "$GLaDOS", tokens: [] },
  nvda: { name: "NVIDIA", tokens: ["NVDA"] },
  spcx: { name: "SpaceX", tokens: ["SPCX"] },
  googl: { name: "Alphabet", tokens: ["GOOGL"] },
  amzn: { name: "Amazon", tokens: ["AMZN"] },
  gme: { name: "GameStop", tokens: ["GME"] },
  spy: { name: "S&P 500", tokens: ["SPY"] },
  chips: { name: "Chips & hardware", tokens: ["NVDA", "AMD", "INTC", "MU", "AVGO", "TSM", "ASML", "MRVL", "SKHY", "DELL"] },
  os: { name: "Operating systems", tokens: ["MSFT", "AAPL", "GOOGL", "IBM", "BB"] },
  index: { name: "Index", tokens: ["SPY", "QQQ"] },
  metals: { name: "Metals & commodities", tokens: ["GLD", "SLV", "USO"] },
};

export const DEFAULT = "glados";

// A miner's word for their choice, made canonical: case-insensitive, and
// anything not on the menu is $GLADOS rather than an error, so a typo never
// costs a payout.
export function rewardOf(x) {
  const k = String(x || "").trim().toLowerCase();
  return Object.hasOwn(MENU, k) ? k : DEFAULT;
}
