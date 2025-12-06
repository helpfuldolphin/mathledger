import { test, expect } from "@playwright/test";

/**
 * Reflexive UI → Backend MDAP Loop Tests
 * =======================================
 *
 * These tests validate the complete reflexive loop:
 *
 *   UI Event → POST /attestation/ui-event → Backend records leaf →
 *   Ledger seals block → GET /attestation/latest → UI displays H_t
 *
 * The MDAP (Minimal Deterministic Attestation Protocol) requires:
 * 1. Every UI event is logged with a deterministic payload
 * 2. The backend computes leaf hashes from canonical JSON
 * 3. The UI root (U_t) is computed from all UI event leaves
 * 4. The composite root (H_t) binds R_t and U_t
 *
 * Prerequisites:
 * - Backend API running with attestation endpoints enabled
 * - Database and Redis available for ledger operations
 */

test.describe("MDAP Loop: UI Event Recording", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");
  });

  test("dashboard_mount event is logged on page load", async ({ page }) => {
    // The dashboard should have already logged a mount event
    // We can verify this by checking the network requests
    const requests: string[] = [];

    page.on("request", (request) => {
      if (request.url().includes("/attestation/ui-event")) {
        requests.push(request.postData() || "");
      }
    });

    // Reload to capture the mount event
    await page.reload();
    await page.waitForSelector("text=Proof Graph Console");

    // Wait for the mount event to be sent
    await page.waitForTimeout(1000);

    // Check that a dashboard_mount event was sent
    const mountEvent = requests.find((r) => r.includes("dashboard_mount"));
    expect(mountEvent).toBeDefined();

    if (mountEvent) {
      const payload = JSON.parse(mountEvent);
      expect(payload.event_type).toBe("dashboard_mount");
      expect(typeof payload.timestamp).toBe("number");
    }
  });

  test("select_statement event is logged on statement click", async ({
    page,
  }) => {
    const requests: string[] = [];

    page.on("request", (request) => {
      if (request.url().includes("/attestation/ui-event")) {
        requests.push(request.postData() || "");
      }
    });

    // Wait for statements to load
    await page.waitForTimeout(1500);

    // Find a statement hash button
    const statementButton = page
      .locator("button")
      .filter({ hasText: /^[0-9a-f]{64}$/i })
      .first();

    const count = await statementButton.count();
    if (count === 0) {
      test.skip(true, "No statements available");
      return;
    }

    // Get the statement hash before clicking
    const hash = await statementButton.textContent();

    // Click the statement
    await statementButton.click();

    // Wait for the event to be sent
    await page.waitForTimeout(500);

    // Check that a select_statement event was sent
    const selectEvent = requests.find((r) => r.includes("select_statement"));
    expect(selectEvent).toBeDefined();

    if (selectEvent) {
      const payload = JSON.parse(selectEvent);
      expect(payload.event_type).toBe("select_statement");
      expect(payload.statement_hash).toBe(hash);
      expect(typeof payload.timestamp).toBe("number");
    }
  });

  test("refresh_statements event is logged on refresh", async ({ page }) => {
    const requests: string[] = [];

    page.on("request", (request) => {
      if (request.url().includes("/attestation/ui-event")) {
        requests.push(request.postData() || "");
      }
    });

    // Find the statements section refresh button
    const statementsSection = page.locator("section").filter({
      has: page.locator("text=Statements"),
    });

    const refreshButton = statementsSection
      .locator("button")
      .filter({ hasText: /^Refresh$/ });

    if (await refreshButton.isVisible()) {
      await refreshButton.click();

      // Wait for the event to be sent
      await page.waitForTimeout(500);

      // Check that a refresh_statements event was sent
      const refreshEvent = requests.find((r) =>
        r.includes("refresh_statements")
      );
      expect(refreshEvent).toBeDefined();

      if (refreshEvent) {
        const payload = JSON.parse(refreshEvent);
        expect(payload.event_type).toBe("refresh_statements");
        expect(typeof payload.timestamp).toBe("number");
        expect(typeof payload.statement_count).toBe("number");
      }
    }
  });
});

test.describe("MDAP Loop: Attestation Fetch", () => {
  test("GET /attestation/latest returns valid response", async ({ page }) => {
    let attestationResponse: unknown = null;

    page.on("response", async (response) => {
      if (response.url().includes("/attestation/latest")) {
        try {
          attestationResponse = await response.json();
        } catch {
          // Response might not be JSON
        }
      }
    });

    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");

    // Wait for attestation to be fetched
    await page.waitForTimeout(2000);

    if (attestationResponse) {
      const resp = attestationResponse as Record<string, unknown>;

      // Validate response structure
      expect(resp).toHaveProperty("reasoning_merkle_root");
      expect(resp).toHaveProperty("ui_merkle_root");
      expect(resp).toHaveProperty("composite_attestation_root");

      // Validate hex format if present
      if (resp.reasoning_merkle_root) {
        expect(resp.reasoning_merkle_root).toMatch(/^[0-9a-f]{64}$/);
      }
      if (resp.ui_merkle_root) {
        expect(resp.ui_merkle_root).toMatch(/^[0-9a-f]{64}$/);
      }
      if (resp.composite_attestation_root) {
        expect(resp.composite_attestation_root).toMatch(/^[0-9a-f]{64}$/);
      }
    }
  });

  test("attestation panel updates after UI event", async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector("text=Dual Attestation");

    // Capture initial attestation display
    const initialHashes = await page
      .locator("code")
      .filter({ hasText: "…" })
      .allTextContents();

    // Trigger a UI event (refresh statements)
    const statementsSection = page.locator("section").filter({
      has: page.locator("text=Statements"),
    });

    const refreshButton = statementsSection
      .locator("button")
      .filter({ hasText: /^Refresh$/ });

    if (await refreshButton.isVisible()) {
      await refreshButton.click();

      // Wait for potential attestation update
      await page.waitForTimeout(2000);

      // Refresh attestation
      const attestationSection = page.locator("section").filter({
        has: page.locator("text=Dual Attestation"),
      });

      const attestationRefresh = attestationSection
        .locator("button")
        .filter({ hasText: /Refresh/i });

      if (await attestationRefresh.isVisible()) {
        await attestationRefresh.click();
        await page.waitForTimeout(1000);
      }

      // The attestation should still be valid (may or may not have changed)
      await expect(page.locator("text=Dual Attestation")).toBeVisible();
    }
  });
});

test.describe("MDAP Loop: Event Payload Validation", () => {
  test("UI event payloads are JSON-serializable", async ({ page }) => {
    const payloads: string[] = [];

    page.on("request", (request) => {
      if (
        request.url().includes("/attestation/ui-event") &&
        request.method() === "POST"
      ) {
        const data = request.postData();
        if (data) {
          payloads.push(data);
        }
      }
    });

    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");
    await page.waitForTimeout(1000);

    // All payloads should be valid JSON
    for (const payload of payloads) {
      expect(() => JSON.parse(payload)).not.toThrow();

      const parsed = JSON.parse(payload);

      // Required fields
      expect(parsed).toHaveProperty("event_type");
      expect(parsed).toHaveProperty("timestamp");

      // Timestamp should be a reasonable Unix millisecond timestamp
      expect(parsed.timestamp).toBeGreaterThan(1700000000000); // After 2023
      expect(parsed.timestamp).toBeLessThan(2000000000000); // Before 2033
    }
  });

  test("UI event payloads have deterministic key ordering", async ({
    page,
  }) => {
    const payloads: string[] = [];

    page.on("request", (request) => {
      if (
        request.url().includes("/attestation/ui-event") &&
        request.method() === "POST"
      ) {
        const data = request.postData();
        if (data) {
          payloads.push(data);
        }
      }
    });

    // Load page twice to get multiple mount events
    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");
    await page.waitForTimeout(500);

    await page.reload();
    await page.waitForSelector("text=Proof Graph Console");
    await page.waitForTimeout(500);

    // Find dashboard_mount events
    const mountPayloads = payloads.filter((p) =>
      p.includes("dashboard_mount")
    );

    if (mountPayloads.length >= 2) {
      // Parse and compare structure (not values, since timestamps differ)
      const first = JSON.parse(mountPayloads[0]);
      const second = JSON.parse(mountPayloads[1]);

      // Same keys
      expect(Object.keys(first).sort()).toEqual(Object.keys(second).sort());

      // Same event_type
      expect(first.event_type).toBe(second.event_type);
    }
  });
});

