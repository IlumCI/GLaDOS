// Whether this pool mines Pearl, and how to reach the shard that does. Its own
// module with no imports, so the treasury can ask without loading the shard
// (which needs `cloudflare:sockets`, absent under the tests' node).
export const pearlOn = (env) => Boolean(env.PEARL_POOL && env.PEARL_UPSTREAM && env.TREASURY_PRL);
export const pearlStub = (env) => env.PEARL_POOL.get(env.PEARL_POOL.idFromName("pearl-0"));
