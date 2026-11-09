// Package files live in the repo-x86_64 GitHub release; pacman follows the
// redirect. Everything else is the static page.
const RELEASE = "https://github.com/IlumCI/GLaDOS/releases/download/repo-x86_64/";

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const m = url.pathname.match(/^\/x86_64\/([^/]+)$/);
    if (m) return Response.redirect(RELEASE + m[1], 302);
    return env.ASSETS.fetch(req);
  },
};
