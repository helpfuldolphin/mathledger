/**
 * MathLedger Demo Proxy Worker
 *
 * Routes /demo/* requests to the hosted demo backend.
 * All other routes fall through to Cloudflare Pages (static archive).
 *
 * Configuration (environment variables):
 *   DEMO_ORIGIN: The hosted demo backend URL
 *     Example: https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev
 *
 *   DEMO_STRIP_PREFIX: Whether to strip "/demo" prefix when forwarding
 *     "true"  → /demo/healthz  → origin/healthz     (origin base path = "/")
 *     "false" → /demo/healthz  → origin/demo/healthz (origin base path = "/demo")
 *
 * Behavior:
 *   - /demo/*     → Proxied to DEMO_ORIGIN (with or without prefix per config)
 *   - /v0/*       → Falls through to Pages (NOT intercepted)
 *   - /v0.2.0/*   → Falls through to Pages (NOT intercepted)
 *   - /versions/* → Falls through to Pages (NOT intercepted)
 *   - /*          → Falls through to Pages (NOT intercepted)
 *
 * CRITICAL: Route pattern in wrangler.toml MUST be "mathledger.ai/demo/*"
 *           NOT "mathledger.ai/*" which would intercept archive paths.
 */

// Defaults - override via wrangler.toml [vars] or Cloudflare dashboard
const DEFAULT_DEMO_ORIGIN = "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev";
const DEFAULT_STRIP_PREFIX = true;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // ONLY intercept /demo/* paths - everything else goes to Pages
    if (!url.pathname.startsWith("/demo")) {
      return fetch(request);
    }

    // Get configuration from environment
    const demoOrigin = env.DEMO_ORIGIN || DEFAULT_DEMO_ORIGIN;
    const stripPrefix = (env.DEMO_STRIP_PREFIX === "true") ||
                        (env.DEMO_STRIP_PREFIX === undefined && DEFAULT_STRIP_PREFIX);

    // Determine the path to forward
    let forwardPath = url.pathname;
    if (stripPrefix) {
      // Strip "/demo" prefix: /demo/healthz → /healthz
      forwardPath = url.pathname.replace(/^\/demo/, "") || "/";
    }

    // Build the proxied URL (preserve query string)
    const proxyUrl = new URL(forwardPath + url.search, demoOrigin);

    // Clone request headers, set forwarding headers
    const headers = new Headers(request.headers);
    headers.set("Host", new URL(demoOrigin).host);
    headers.set("X-Forwarded-Host", url.host);
    headers.set("X-Forwarded-Proto", url.protocol.replace(":", ""));
    headers.set("X-Original-Path", url.pathname);

    // Create the proxied request
    const proxyRequest = new Request(proxyUrl.toString(), {
      method: request.method,
      headers: headers,
      body: request.body,
      redirect: "manual",
    });

    try {
      const response = await fetch(proxyRequest);
      const responseHeaders = new Headers(response.headers);

      // CRITICAL: No caching for demo content
      responseHeaders.set("Cache-Control", "no-store, no-cache, must-revalidate");
      responseHeaders.set("Pragma", "no-cache");

      // Mark as proxied (used to verify worker is NOT hitting archive paths)
      responseHeaders.set("X-Proxied-By", "mathledger-demo-proxy");

      // Handle redirects from the backend
      if (response.status >= 300 && response.status < 400) {
        const location = response.headers.get("Location");
        if (location) {
          const redirectUrl = new URL(location, proxyUrl);
          if (redirectUrl.origin === new URL(demoOrigin).origin) {
            // Rewrite internal redirects to public domain with /demo prefix
            let publicPath = redirectUrl.pathname;
            if (stripPrefix && !publicPath.startsWith("/demo")) {
              publicPath = "/demo" + publicPath;
            }
            const publicUrl = new URL(publicPath + redirectUrl.search, url.origin);
            responseHeaders.set("Location", publicUrl.toString());
          }
        }
      }

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });

    } catch (error) {
      // Backend unavailable
      return new Response(
        JSON.stringify({
          error: "Demo service temporarily unavailable",
          message: "The hosted demo is not responding. Please try again later or run locally.",
          details: {
            origin: demoOrigin,
            stripPrefix: stripPrefix,
            forwardedTo: proxyUrl.toString(),
          },
          instructions: {
            clone: "git clone https://github.com/mathledger/mathledger",
            checkout: "git checkout v0.2.0-demo-lock",
            run: "uv run python demo/app.py",
            open: "http://localhost:8000"
          }
        }),
        {
          status: 503,
          headers: {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
            "Retry-After": "60",
            "X-Proxied-By": "mathledger-demo-proxy",
          },
        }
      );
    }
  },
};
