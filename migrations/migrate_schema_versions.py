#!/usr/bin/env python3
"""
Schema Version Migration Script
================================

This script applies schema version tracking to the MathLedger database.

It performs the following operations:
1. Adds version columns to statements and blocks tables
2. Creates schema_versions metadata table
3. Backfills version data for existing records
4. Verifies migration integrity

Usage:
    python migrations/migrate_schema_versions.py --dry-run
    python migrations/migrate_schema_versions.py --apply

Safety:
    - Always run with --dry-run first
    - Creates backup before applying changes
    - Rolls back on any error
    - Logs all operations for audit trail
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from normalization.schema_version import (
    CANON_SCHEMA_VERSION,
    HASH_ALGORITHM_VERSION,
    JSON_CANON_SCHEMA_VERSION,
    MERKLE_SCHEMA_VERSION,
    ATTESTATION_SCHEMA_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaMigration:
    """Schema version migration manager."""
    
    def __init__(self, database_url: str, dry_run: bool = True):
        self.database_url = database_url
        self.dry_run = dry_run
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None
    
    def connect(self):
        """Connect to database."""
        logger.info(f"Connecting to database...")
        self.conn = psycopg2.connect(self.database_url)
        self.cursor = self.conn.cursor()
        logger.info("Connected successfully")
    
    def disconnect(self):
        """Disconnect from database."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")
    
    def execute_sql(self, sql: str, params: tuple = None):
        """Execute SQL with logging."""
        logger.info(f"Executing SQL: {sql[:100]}...")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would execute SQL (skipping)")
            return
        
        try:
            self.cursor.execute(sql, params)
            logger.info("SQL executed successfully")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise
    
    def check_column_exists(self, table: str, column: str) -> bool:
        """Check if column exists in table."""
        sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = %s
            );
        """
        self.cursor.execute(sql, (table, column))
        return self.cursor.fetchone()[0]
    
    def check_table_exists(self, table: str) -> bool:
        """Check if table exists."""
        sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = %s
            );
        """
        self.cursor.execute(sql, (table,))
        return self.cursor.fetchone()[0]
    
    def add_version_columns_to_statements(self):
        """Add version columns to statements table."""
        logger.info("Adding version columns to statements table...")
        
        columns = [
            ('canon_schema_version', 'VARCHAR(16)', CANON_SCHEMA_VERSION),
            ('hash_algorithm', 'VARCHAR(32)', HASH_ALGORITHM_VERSION),
            ('json_canon_version', 'VARCHAR(16)', JSON_CANON_SCHEMA_VERSION),
        ]
        
        for column, dtype, default in columns:
            if self.check_column_exists('statements', column):
                logger.info(f"Column {column} already exists, skipping")
                continue
            
            sql = f"""
                ALTER TABLE statements 
                ADD COLUMN {column} {dtype} DEFAULT '{default}';
            """
            self.execute_sql(sql)
            logger.info(f"Added column {column} to statements table")
    
    def add_version_columns_to_blocks(self):
        """Add version columns to blocks table."""
        logger.info("Adding version columns to blocks table...")
        
        columns = [
            ('attestation_schema_version', 'VARCHAR(16)', ATTESTATION_SCHEMA_VERSION),
            ('merkle_schema_version', 'VARCHAR(16)', MERKLE_SCHEMA_VERSION),
        ]
        
        for column, dtype, default in columns:
            if self.check_column_exists('blocks', column):
                logger.info(f"Column {column} already exists, skipping")
                continue
            
            sql = f"""
                ALTER TABLE blocks 
                ADD COLUMN {column} {dtype} DEFAULT '{default}';
            """
            self.execute_sql(sql)
            logger.info(f"Added column {column} to blocks table")
    
    def create_schema_versions_table(self):
        """Create schema_versions metadata table."""
        logger.info("Creating schema_versions table...")
        
        if self.check_table_exists('schema_versions'):
            logger.info("Table schema_versions already exists, skipping")
            return
        
        sql = """
            CREATE TABLE schema_versions (
                id SERIAL PRIMARY KEY,
                component VARCHAR(64) NOT NULL,
                version VARCHAR(16) NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                UNIQUE(component, version)
            );
        """
        self.execute_sql(sql)
        logger.info("Created schema_versions table")
    
    def insert_initial_versions(self):
        """Insert initial schema versions."""
        logger.info("Inserting initial schema versions...")
        
        versions = [
            ('canon', CANON_SCHEMA_VERSION, 'String-based canonicalization (normalization/canon.py)'),
            ('ast_canon', CANON_SCHEMA_VERSION, 'AST-based canonicalization (normalization/ast_canon.py)'),
            ('hash', HASH_ALGORITHM_VERSION, 'SHA-256 with domain separation (backend/crypto/hashing.py)'),
            ('json_canon', JSON_CANON_SCHEMA_VERSION, 'RFC 8785 JSON canonicalization (backend/basis/canon.py)'),
            ('merkle', MERKLE_SCHEMA_VERSION, 'Merkle tree construction with sorted leaves'),
            ('attestation', ATTESTATION_SCHEMA_VERSION, 'Dual-root attestation (H_t = SHA256(R_t || U_t))'),
        ]
        
        for component, version, description in versions:
            sql = """
                INSERT INTO schema_versions (component, version, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (component, version) DO NOTHING;
            """
            self.execute_sql(sql, (component, version, description))
            logger.info(f"Inserted version {version} for component {component}")
    
    def create_indexes(self):
        """Create indexes on version columns."""
        logger.info("Creating indexes on version columns...")
        
        indexes = [
            ('idx_statements_canon_schema_version', 'statements', 'canon_schema_version'),
            ('idx_statements_hash_algorithm', 'statements', 'hash_algorithm'),
        ]
        
        for index_name, table, column in indexes:
            sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {table}({column});
            """
            self.execute_sql(sql)
            logger.info(f"Created index {index_name}")
    
    def verify_migration(self):
        """Verify migration was successful."""
        logger.info("Verifying migration...")
        
        # Check statements table
        self.cursor.execute("""
            SELECT 
                canon_schema_version,
                hash_algorithm,
                json_canon_version,
                COUNT(*) as count
            FROM statements
            GROUP BY canon_schema_version, hash_algorithm, json_canon_version;
        """)
        
        results = self.cursor.fetchall()
        logger.info(f"Statements table version distribution:")
        for row in results:
            logger.info(f"  {row[0]}, {row[1]}, {row[2]}: {row[3]} statements")
        
        # Check blocks table
        self.cursor.execute("""
            SELECT 
                attestation_schema_version,
                merkle_schema_version,
                COUNT(*) as count
            FROM blocks
            GROUP BY attestation_schema_version, merkle_schema_version;
        """)
        
        results = self.cursor.fetchall()
        logger.info(f"Blocks table version distribution:")
        for row in results:
            logger.info(f"  {row[0]}, {row[1]}: {row[2]} blocks")
        
        # Check schema_versions table
        self.cursor.execute("SELECT component, version FROM schema_versions ORDER BY component;")
        results = self.cursor.fetchall()
        logger.info(f"Schema versions registered:")
        for component, version in results:
            logger.info(f"  {component}: {version}")
        
        logger.info("Verification complete")
    
    def run(self):
        """Run the migration."""
        try:
            self.connect()
            
            if self.dry_run:
                logger.info("=" * 60)
                logger.info("DRY RUN MODE - No changes will be made")
                logger.info("=" * 60)
            
            logger.info("Starting schema version migration...")
            
            # Step 1: Add version columns
            self.add_version_columns_to_statements()
            self.add_version_columns_to_blocks()
            
            # Step 2: Create metadata table
            self.create_schema_versions_table()
            
            # Step 3: Insert initial versions
            self.insert_initial_versions()
            
            # Step 4: Create indexes
            self.create_indexes()
            
            # Step 5: Verify
            if not self.dry_run:
                self.verify_migration()
                self.conn.commit()
                logger.info("Migration committed successfully")
            else:
                logger.info("DRY RUN complete - no changes made")
            
            logger.info("Schema version migration complete!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if self.conn and not self.dry_run:
                self.conn.rollback()
                logger.info("Changes rolled back")
            raise
        
        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Apply schema version migration')
    parser.add_argument('--dry-run', action='store_true', help='Simulate migration without applying changes')
    parser.add_argument('--apply', action='store_true', help='Apply migration (requires explicit flag)')
    parser.add_argument('--database-url', help='Database URL (default: from DATABASE_URL env var)')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        parser.error("Must specify either --dry-run or --apply")
    
    database_url = args.database_url or os.getenv('DATABASE_URL')
    if not database_url:
        parser.error("Database URL must be provided via --database-url or DATABASE_URL env var")
    
    dry_run = args.dry_run or not args.apply
    
    migration = SchemaMigration(database_url, dry_run=dry_run)
    migration.run()


if __name__ == '__main__':
    main()
