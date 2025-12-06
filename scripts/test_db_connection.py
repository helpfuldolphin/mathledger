#!/usr/bin/env python3
"""
Quick database connection test script.

Tests connectivity to PostgreSQL using DATABASE_URL from environment.
Provides clear error messages for SSL and other connection issues.

Usage:
    uv run python scripts/test_db_connection.py

Environment:
    DATABASE_URL: PostgreSQL connection URL (required)
"""

import os
import sys
from urllib.parse import urlparse, parse_qs

try:
    import psycopg
except ImportError:
    print("ERROR: psycopg package not installed")
    print("  Install with: uv add psycopg[binary]")
    sys.exit(1)


def mask_password(url: str) -> str:
    """Mask password in URL for safe display."""
    if not url:
        return ""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def check_sslmode(url: str) -> tuple:
    """
    Check if sslmode is present in DATABASE_URL.
    
    Returns:
        (has_sslmode, recommendation)
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    
    if "sslmode" in query:
        sslmode = query["sslmode"][0]
        return True, f"Current: sslmode={sslmode}"
    
    # Determine recommendation based on host
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1") or not host:
        return False, "Recommendation: add ?sslmode=disable for local Docker"
    else:
        return False, "Recommendation: add ?sslmode=require for remote database"


def main():
    """Test database connection and report results."""
    db_url = os.environ.get("DATABASE_URL")
    
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("  Set it with: export DATABASE_URL='postgresql://user:pass@host:port/dbname?sslmode=disable'")
        print("  Or load from config: export $(grep -v '^#' config/first_organism.env | xargs)")
        sys.exit(1)
    
    db_url_masked = mask_password(db_url)
    print(f"Testing connection to: {db_url_masked}")
    
    # Check sslmode presence
    has_sslmode, sslmode_info = check_sslmode(db_url)
    if not has_sslmode:
        print(f"WARNING: {sslmode_info}")
    
    # Attempt connection
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result[0] == 1:
                    print("✓ SUCCESS: Database connection successful")
                    print(f"  Connection info: {sslmode_info}")
                    sys.exit(0)
                else:
                    print("ERROR: Unexpected query result")
                    sys.exit(1)
    except psycopg.OperationalError as e:
        error_str = str(e).lower()
        
        # Check for SSL negotiation errors
        if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
            print("✗ FAILED: SSL negotiation error")
            print(f"  Error: {e}")
            print(f"  {sslmode_info}")
            print("\n  Troubleshooting:")
            print("  - For local Docker Postgres: add ?sslmode=disable to DATABASE_URL")
            print("  - For remote Postgres with SSL: add ?sslmode=require to DATABASE_URL")
            print("  - See docs/FIRST_ORGANISM_ENV.md for more details")
            sys.exit(1)
        
        # Check for authentication errors
        elif "password" in error_str or "authentication" in error_str:
            print("✗ FAILED: Authentication error")
            print(f"  Error: {e}")
            print("  Check username and password in DATABASE_URL")
            sys.exit(1)
        
        # Check for timeout errors
        elif "timeout" in error_str or "timed out" in error_str:
            print("✗ FAILED: Connection timeout")
            print(f"  Error: {e}")
            print("  Check if database is running and accessible")
            sys.exit(1)
        
        # Generic operational error
        else:
            print("✗ FAILED: Database connection error")
            print(f"  Error: {e}")
            print(f"  {sslmode_info}")
            sys.exit(1)
    
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for SSL errors in generic exceptions too
        if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
            print("✗ FAILED: SSL negotiation error")
            print(f"  Error: {e}")
            print(f"  {sslmode_info}")
            print("\n  Troubleshooting:")
            print("  - For local Docker Postgres: add ?sslmode=disable to DATABASE_URL")
            print("  - For remote Postgres with SSL: add ?sslmode=require to DATABASE_URL")
            print("  - See docs/FIRST_ORGANISM_ENV.md for more details")
            sys.exit(1)
        
        print("✗ FAILED: Unexpected error")
        print(f"  Error: {e}")
        print(f"  Type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()

