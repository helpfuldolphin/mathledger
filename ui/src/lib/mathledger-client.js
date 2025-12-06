/**
 * MathLedger Client SDK
 * 
 * Provides optimized communication between Node/UI and FastAPI backend
 * with latency tracking and error handling.
 */

class MathLedgerClient {
  /**
   * Create a new MathLedger client.
   * 
   * @param {string} baseUrl - Base URL of the FastAPI backend
   * @param {Object} options - Configuration options
   * @param {string} options.apiKey - API key for authentication
   * @param {number} options.timeout - Request timeout in milliseconds (default: 5000)
   * @param {boolean} options.trackLatency - Enable latency tracking (default: true)
   */
  constructor(baseUrl, options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey || 'devkey';
    this.timeout = options.timeout || 5000;
    this.trackLatency = options.trackLatency !== false;
    this.latencyMeasurements = [];
  }

  /**
   * Make an HTTP request with latency tracking.
   * 
   * @private
   * @param {string} method - HTTP method
   * @param {string} path - Request path
   * @param {Object} options - Request options
   * @returns {Promise<Object>} Response data
   */
  async _request(method, path, options = {}) {
    const startTime = performance.now();
    const url = `${this.baseUrl}${path}`;
    
    const headers = {
      'Content-Type': 'application/json',
      'X-API-Key': this.apiKey,
      ...options.headers
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const endTime = performance.now();
      const duration = endTime - startTime;

      if (this.trackLatency) {
        this._recordLatency(method, path, duration, response.ok);
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      const endTime = performance.now();
      const duration = endTime - startTime;

      if (this.trackLatency) {
        this._recordLatency(method, path, duration, false, error.message);
      }

      throw error;
    }
  }

  /**
   * Record latency measurement.
   * 
   * @private
   */
  _recordLatency(method, path, duration, success, error = null) {
    this.latencyMeasurements.push({
      operation: `${method} ${path}`,
      duration_ms: duration,
      success,
      error,
      timestamp: new Date().toISOString()
    });

    if (this.latencyMeasurements.length > 1000) {
      this.latencyMeasurements.shift();
    }
  }

  /**
   * Get system metrics.
   * 
   * @returns {Promise<Object>} System metrics
   */
  async getMetrics() {
    return this._request('GET', '/metrics');
  }

  /**
   * Get latest block.
   * 
   * @returns {Promise<Object>} Latest block data
   */
  async getLatestBlock() {
    return this._request('GET', '/blocks/latest');
  }

  /**
   * Query statements.
   * 
   * @param {Object} params - Query parameters
   * @param {string} params.hash - Statement hash (optional)
   * @param {number} params.limit - Result limit (optional)
   * @returns {Promise<Array>} Array of statements
   */
  async getStatements(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const path = `/statements${queryString ? '?' + queryString : ''}`;
    return this._request('GET', path);
  }

  /**
   * Get statement by hash.
   * 
   * @param {string} hash - Statement hash
   * @returns {Promise<Object>} Statement data
   */
  async getStatement(hash) {
    return this._request('GET', `/statements?hash=${hash}`);
  }

  /**
   * Get health status.
   * 
   * @returns {Promise<Object>} Health status
   */
  async getHealth() {
    return this._request('GET', '/health');
  }

  /**
   * Get heartbeat data.
   * 
   * @returns {Promise<Object>} Heartbeat data
   */
  async getHeartbeat() {
    return this._request('GET', '/heartbeat.json');
  }

  /**
   * Get latency statistics.
   * 
   * @returns {Object} Latency statistics
   */
  getLatencyStats() {
    if (this.latencyMeasurements.length === 0) {
      return {
        count: 0,
        mean_ms: 0,
        min_ms: 0,
        max_ms: 0,
        p50_ms: 0,
        p95_ms: 0,
        p99_ms: 0,
        success_rate: 0
      };
    }

    const durations = this.latencyMeasurements
      .map(m => m.duration_ms)
      .sort((a, b) => a - b);

    const successes = this.latencyMeasurements.filter(m => m.success).length;

    const percentile = (arr, p) => {
      const index = Math.ceil(arr.length * p) - 1;
      return arr[Math.max(0, index)];
    };

    return {
      count: this.latencyMeasurements.length,
      mean_ms: durations.reduce((a, b) => a + b, 0) / durations.length,
      min_ms: Math.min(...durations),
      max_ms: Math.max(...durations),
      p50_ms: percentile(durations, 0.50),
      p95_ms: percentile(durations, 0.95),
      p99_ms: percentile(durations, 0.99),
      success_rate: (successes / this.latencyMeasurements.length) * 100
    };
  }

  /**
   * Clear latency measurements.
   */
  clearLatencyStats() {
    this.latencyMeasurements = [];
  }

  /**
   * Check if latency target is met (<200ms).
   * 
   * @returns {boolean} True if target is met
   */
  isLatencyTargetMet() {
    const stats = this.getLatencyStats();
    return stats.p95_ms < 200;
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MathLedgerClient;
}

export default MathLedgerClient;
