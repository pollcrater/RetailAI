"""
Multi-File Schema Analyzer & Comparison Tool
Analyzes multiple CSV files, detects schema variations, and suggests unified data model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
from collections import defaultdict
import warnings
import argparse
import re
warnings.filterwarnings('ignore')

class SchemaAnalyzer:
    """Analyze and compare schemas across multiple CSV files"""
    
    def __init__(self, *, model_mode: str = "separate"):
        if model_mode not in {"separate", "by_type"}:
            raise ValueError("model_mode must be one of: 'separate', 'by_type'")
        self.model_mode = model_mode
        self.files_analyzed = []
        self.schemas = {}
        self.comparisons = {}

    def _sanitize_identifier(self, name: str) -> str:
        """Sanitize a string into a DuckDB-safe identifier.

        Rules:
        - lowercase
        - replace non-alphanumerics with '_'
        - collapse repeated '_'
        - ensure it doesn't start with a digit
        """
        if name is None:
            name = ""
        cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
        cleaned = re.sub(r"_+", "_", cleaned)
        if not cleaned:
            cleaned = "col"
        if cleaned[0].isdigit():
            cleaned = f"c_{cleaned}"
        return cleaned

    def _make_unique_identifiers(self, names: List[str]) -> Dict[str, str]:
        """Map original names to unique sanitized identifiers, de-duping with suffixes."""
        mapping: Dict[str, str] = {}
        used: Dict[str, int] = {}
        for original in names:
            base = self._sanitize_identifier(original)
            if base not in used:
                used[base] = 1
                mapping[original] = base
                continue
            used[base] += 1
            mapping[original] = f"{base}_{used[base]}"
        return mapping
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze single CSV file schema"""
        print(f"\n{'='*80}")
        print(f"📄 Analyzing: {Path(file_path).name}")
        print(f"{'='*80}")

        file_path = str(file_path)
        if not Path(file_path).exists():
            raise FileNotFoundError(f"No such file: {file_path}")

        def _count_rows_chunked(path: str) -> int:
            total = 0
            for chunk in pd.read_csv(path, low_memory=False, chunksize=200_000):
                total += len(chunk)
            return total
        
        # Load file
        df = pd.read_csv(file_path, low_memory=False, nrows=1000)  # Sample first 1000 rows for speed
        
        original_cols = list(df.columns)
        col_mapping = self._make_unique_identifiers(original_cols)

        file_info = {
            'filename': Path(file_path).name,
            'filepath': file_path,
            'row_count': _count_rows_chunked(file_path),
            'sample_rows': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'column_name_map': col_mapping,  # original -> sanitized
            'file_type': None,
            'business_domain': None,
            'key_entities': []
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = self._analyze_column(df[col], col)
            file_info['columns'][col] = col_analysis
        
        # Detect file type and business domain
        file_info['file_type'] = self._detect_file_type(df, file_info)
        file_info['business_domain'] = self._detect_business_domain(df, file_info)
        file_info['key_entities'] = self._identify_key_entities(df, file_info)
        
        # Store schema
        self.schemas[file_info['filename']] = file_info
        self.files_analyzed.append(file_info['filename'])
        
        # Print summary
        print(f"  📊 Rows: {file_info['row_count']:,}")
        print(f"  📋 Columns: {file_info['column_count']}")
        print(f"  🏷️  File Type: {file_info['file_type']}")
        print(f"  🎯 Domain: {file_info['business_domain']}")
        print(f"  🔑 Key Entities: {', '.join(file_info['key_entities'])}")
        
        return file_info
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict:
        """Detailed column analysis"""
        analysis = {
            'name': col_name,
            'dtype': str(series.dtype),
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().sum() / len(series) * 100),
            'unique_count': int(series.nunique()),
            'unique_percentage': float(series.nunique() / len(series) * 100),
            'sample_values': series.dropna().head(5).tolist(),
            'inferred_type': None,
            'business_meaning': None
        }
        
        # Infer semantic type
        analysis['inferred_type'] = self._infer_column_type(series, col_name)
        analysis['business_meaning'] = self._infer_business_meaning(series, col_name, analysis)
        
        return analysis
    
    def _infer_column_type(self, series: pd.Series, col_name: str) -> str:
        """Infer semantic column type"""
        col_lower = col_name.lower()
        
        # Check for ID columns
        if 'id' in col_lower or 'index' in col_lower:
            if series.nunique() == len(series):
                return 'unique_identifier'
            return 'identifier'
        
        # Check for date columns
        if any(kw in col_lower for kw in ['date', 'time', 'month', 'year']):
            return 'temporal'
        
        # Check for numeric
        if series.dtype in ['int64', 'float64']:
            # Check if it's a price/amount
            if any(kw in col_lower for kw in ['price', 'mrp', 'rate', 'amount', 'total', 'tp']):
                return 'monetary'
            # Check if it's a quantity
            if any(kw in col_lower for kw in ['qty', 'quantity', 'pcs', 'weight']):
                return 'quantity'
            # Check if it's a count
            if series.min() >= 0 and series.max() < 10000:
                return 'count'
            return 'numeric'
        
        # Check for categorical
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05:
                return 'categorical_low'
            elif unique_ratio < 0.5:
                return 'categorical_high'
            else:
                return 'text'
        
        return 'unknown'
    
    def _infer_business_meaning(self, series: pd.Series, col_name: str, analysis: Dict) -> str:
        """Infer business meaning of column"""
        col_lower = col_name.lower()
        
        # Product/Catalog related
        if any(kw in col_lower for kw in ['sku', 'product', 'style', 'catalog', 'item']):
            return 'product_identifier'
        if any(kw in col_lower for kw in ['category', 'type', 'class']):
            return 'product_category'
        
        # Customer related
        if 'customer' in col_lower or 'client' in col_lower:
            return 'customer_identifier'
        
        # Financial
        if any(kw in col_lower for kw in ['mrp', 'price', 'rate']):
            return 'pricing'
        if any(kw in col_lower for kw in ['amount', 'total', 'gross']):
            return 'transaction_amount'
        
        # Logistics
        if 'weight' in col_lower:
            return 'logistics_weight'
        if 'size' in col_lower:
            return 'product_size'
        
        # Channel/Platform
        if any(kw in col_lower for kw in ['amazon', 'flipkart', 'myntra', 'ajio', 'paytm', 'snapdeal']):
            return 'sales_channel_price'
        
        # Temporal
        if any(kw in col_lower for kw in ['date', 'month', 'year']):
            return 'temporal_dimension'
        
        return 'general'
    
    def _detect_file_type(self, df: pd.DataFrame, file_info: Dict) -> str:
        """Detect the type of file (transactional, catalog, etc.)"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Transactional data indicators
        transaction_indicators = sum([
            'customer' in ' '.join(columns_lower),
            'date' in ' '.join(columns_lower),
            any(kw in ' '.join(columns_lower) for kw in ['amount', 'total', 'gross']),
            any(kw in ' '.join(columns_lower) for kw in ['quantity', 'pcs']),
        ])
        
        # Catalog/Product data indicators
        catalog_indicators = sum([
            any(kw in ' '.join(columns_lower) for kw in ['sku', 'style', 'catalog']),
            any(kw in ' '.join(columns_lower) for kw in ['mrp', 'price']),
            any(kw in ' '.join(columns_lower) for kw in ['category', 'type']),
            sum(1 for col in columns_lower if any(platform in col for platform in ['amazon', 'flipkart', 'myntra'])) >= 2
        ])
        
        if transaction_indicators >= 3:
            return 'TRANSACTIONAL'
        elif catalog_indicators >= 3:
            return 'PRODUCT_CATALOG'
        else:
            return 'UNKNOWN'
    
    def _detect_business_domain(self, df: pd.DataFrame, file_info: Dict) -> str:
        """Detect business domain"""
        if file_info['file_type'] == 'TRANSACTIONAL':
            return 'Sales Transactions'
        elif file_info['file_type'] == 'PRODUCT_CATALOG':
            return 'Product Master Data'
        return 'Unknown'
    
    def _identify_key_entities(self, df: pd.DataFrame, file_info: Dict) -> List[str]:
        """Identify key business entities in the data"""
        entities = []
        columns_lower = [col.lower() for col in df.columns]
        
        if any('customer' in col for col in columns_lower):
            entities.append('Customer')
        if any('product' in col or 'style' in col or 'sku' in col for col in columns_lower):
            entities.append('Product')
        if any('date' in col or 'month' in col for col in columns_lower):
            entities.append('Time')
        if any('amount' in col or 'total' in col or 'gross' in col for col in columns_lower):
            entities.append('Transaction')
        if any(platform in ' '.join(columns_lower) for platform in ['amazon', 'flipkart', 'myntra']):
            entities.append('Sales_Channel')
        
        return entities
    
    def compare_schemas(self) -> Dict[str, Any]:
        """Compare all analyzed schemas"""
        if len(self.schemas) < 2:
            return {'error': 'Need at least 2 files to compare'}
        
        print(f"\n{'='*80}")
        print(f"🔍 SCHEMA COMPARISON ANALYSIS")
        print(f"{'='*80}")
        
        comparison = {
            'files_compared': len(self.schemas),
            'file_types': {},
            'common_columns': [],
            'unique_columns': {},
            'schema_compatibility': None,
            'unification_strategy': None,
            'recommendations': []
        }
        
        # Group files by type
        for filename, schema in self.schemas.items():
            file_type = schema['file_type']
            if file_type not in comparison['file_types']:
                comparison['file_types'][file_type] = []
            comparison['file_types'][file_type].append(filename)
        
        print(f"\n📂 File Types Detected:")
        for file_type, files in comparison['file_types'].items():
            print(f"  • {file_type}: {len(files)} file(s)")
            for f in files:
                print(f"    - {f}")
        
        # Find common columns across all files
        all_columns = [set(schema['columns'].keys()) for schema in self.schemas.values()]
        comparison['common_columns'] = list(set.intersection(*all_columns))
        
        # Find unique columns per file
        for filename, schema in self.schemas.items():
            unique_cols = set(schema['columns'].keys())
            for other_file, other_schema in self.schemas.items():
                if other_file != filename:
                    unique_cols -= set(other_schema['columns'].keys())
            comparison['unique_columns'][filename] = list(unique_cols)
        
        print(f"\n🔗 Common Columns Across All Files: {len(comparison['common_columns'])}")
        if comparison['common_columns']:
            for col in comparison['common_columns']:
                print(f"  • {col}")
        else:
            print("  ⚠️  No common columns found!")
        
        print(f"\n🆕 Unique Columns Per File:")
        for filename, unique_cols in comparison['unique_columns'].items():
            print(f"  {filename}: {len(unique_cols)} unique column(s)")
            if unique_cols:
                for col in unique_cols[:5]:  # Show first 5
                    print(f"    - {col}")
                if len(unique_cols) > 5:
                    print(f"    ... and {len(unique_cols) - 5} more")
        
        # Determine schema compatibility
        comparison['schema_compatibility'] = self._assess_compatibility()
        
        # Suggest unification strategy
        comparison['unification_strategy'] = self._suggest_unification_strategy()
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_recommendations()
        
        self.comparisons = comparison
        return comparison
    
    def _assess_compatibility(self) -> str:
        """Assess if schemas can be unified"""
        file_types = set(schema['file_type'] for schema in self.schemas.values())
        
        if len(file_types) == 1:
            return "COMPATIBLE - Same file type"
        elif len(file_types) > 1 and 'UNKNOWN' not in file_types:
            return "INCOMPATIBLE - Different file types (requires separate tables)"
        else:
            return "PARTIAL - Manual inspection required"
    
    def _suggest_unification_strategy(self) -> str:
        """Suggest strategy for handling multiple files"""
        file_types = set(schema['file_type'] for schema in self.schemas.values())
        
        if len(file_types) == 1:
            return "VERTICAL_UNION - Stack all files with same schema"
        else:
            return "SEPARATE_TABLES - Create different tables for different file types, link via foreign keys"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        file_types = set(schema['file_type'] for schema in self.schemas.values())
        
        if 'TRANSACTIONAL' in file_types and 'PRODUCT_CATALOG' in file_types:
            recs.append("Create a STAR SCHEMA with fact table (transactions) and dimension table (products)")
            recs.append("Link transactions to product catalog via SKU/Product ID")
            recs.append("Consider creating a unified product master from catalog data")
        
        if len(file_types) == 1 and file_types.pop() == 'TRANSACTIONAL':
            recs.append("All files are transactional - can be combined into single fact table")
            recs.append("Add a 'source_file' column to track data origin")
            recs.append("Standardize column names across all files before union")
        
        if len(self.schemas) > 3:
            recs.append("Consider partitioning data by date/month for better query performance")
            recs.append("Create aggregated summary tables for faster reporting")
        
        # Check for missing key columns
        for filename, schema in self.schemas.items():
            if schema['file_type'] == 'TRANSACTIONAL':
                cols_lower = [col.lower() for col in schema['columns'].keys()]
                if not any('date' in col for col in cols_lower):
                    recs.append(f"⚠️  {filename}: Missing date/time column - may limit temporal analysis")
                if not any('customer' in col for col in cols_lower):
                    recs.append(f"⚠️  {filename}: Missing customer identifier - limits customer analytics")
        
        return recs
    
    def generate_unified_schema(self) -> Dict[str, Any]:
        """Generate data model tables.

        Modes:
        - separate: one table per file (no schema mixing)
        - by_type: previous behavior (union columns by inferred file_type)
        """
        print(f"\n{'='*80}")
        if self.model_mode == "separate":
            print(f"🗄️  DATA MODEL GENERATION (SEPARATE TABLES PER FILE)")
        else:
            print(f"🗄️  DATA MODEL GENERATION (UNION BY FILE TYPE)")
        print(f"{'='*80}")
        
        unified = {
            'tables': {},
            'relationships': [],
            'sql_ddl': []
        }
        
        if self.model_mode == "separate":
            # One table per file
            for filename, schema in self.schemas.items():
                table_name = f"raw_{self._sanitize_identifier(Path(filename).stem)}"
                unified['tables'][table_name] = {
                    'mode': 'separate',
                    'source_files': [schema['filename']],
                    'file_type': schema['file_type'],
                    'columns': schema['columns'],
                    'column_name_map': schema.get('column_name_map', {})
                }
                ddl = self._generate_ddl(table_name, schema['columns'])
                unified['sql_ddl'].append(ddl)
        else:
            file_types: Dict[str, List[Dict[str, Any]]] = {}
            for _filename, schema in self.schemas.items():
                file_type = schema['file_type']
                file_types.setdefault(file_type, []).append(schema)

            # Create tables for each file type (union columns across files of same type)
            for file_type, schemas in file_types.items():
                table_name = self._get_table_name(file_type)

                # Collect all columns across files of same type
                all_columns: Dict[str, Any] = {}
                for schema in schemas:
                    for col_name, col_info in schema['columns'].items():
                        if col_name not in all_columns:
                            all_columns[col_name] = col_info
                        # If column appears in multiple files, track it
                        elif 'appears_in_files' not in all_columns[col_name]:
                            all_columns[col_name]['appears_in_files'] = [schema['filename']]
                        else:
                            all_columns[col_name]['appears_in_files'].append(schema['filename'])

                unified['tables'][table_name] = {
                    'mode': 'by_type',
                    'file_type': file_type,
                    'source_files': [s['filename'] for s in schemas],
                    'columns': all_columns
                }

                # Generate SQL DDL
                ddl = self._generate_ddl(table_name, all_columns)
                unified['sql_ddl'].append(ddl)
        
        # Define relationships (only meaningful for by_type mode)
        if self.model_mode == "by_type":
            if 'sales_transactions' in unified['tables'] and 'product_catalog' in unified['tables']:
                unified['relationships'].append({
                    'from_table': 'sales_transactions',
                    'to_table': 'product_catalog',
                    'join_type': 'LEFT JOIN',
                    'on_columns': ['SKU', 'STYLE', 'PRODUCT_ID'],
                    'description': 'Link transactions to product details'
                })
        
        # Print schema
        print(f"\n📊 Data Model:")
        print(f"  Tables: {len(unified['tables'])}")
        for table_name, table_info in unified['tables'].items():
            print(f"\n  📋 {table_name.upper()}")
            if 'file_type' in table_info:
                print(f"     Type: {table_info['file_type']}")
            print(f"     Sources: {', '.join(table_info.get('source_files', []))}")
            print(f"     Columns: {len(table_info['columns'])}")
        
        if unified['relationships']:
            print(f"\n🔗 Relationships:")
            for rel in unified['relationships']:
                print(f"  {rel['from_table']} → {rel['to_table']} ({rel['join_type']})")
        
        return unified
    
    def _get_table_name(self, file_type: str) -> str:
        """Get appropriate table name for file type"""
        mapping = {
            'TRANSACTIONAL': 'sales_transactions',
            'PRODUCT_CATALOG': 'product_catalog',
            'UNKNOWN': 'misc_data'
        }
        return mapping.get(file_type, 'data_table')
    
    def _generate_ddl(self, table_name: str, columns: Dict) -> str:
        """Generate SQL DDL for table"""
        ddl = f"CREATE TABLE {table_name} (\n"
        
        col_definitions = []
        for col_name, col_info in columns.items():
            sql_type = self._get_sql_type(col_info)
            col_def = f"    {col_name.lower().replace(' ', '_')} {sql_type}"
            col_definitions.append(col_def)
        
        # Add metadata columns
        col_definitions.append("    source_file VARCHAR(255)")
        col_definitions.append("    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        ddl += ",\n".join(col_definitions)
        ddl += "\n);"
        
        return ddl
    
    def _get_sql_type(self, col_info: Dict) -> str:
        """Map Python dtype to SQL type"""
        dtype = col_info['dtype']
        inferred = col_info.get('inferred_type', '')
        
        if dtype == 'int64':
            return 'INTEGER'
        elif dtype == 'float64':
            if inferred == 'monetary':
                return 'DECIMAL(15,2)'
            return 'DOUBLE'
        elif dtype == 'object':
            if inferred == 'temporal':
                return 'DATE'
            elif col_info['unique_percentage'] > 90:
                return 'TEXT'
            else:
                return 'VARCHAR(255)'
        else:
            return 'VARCHAR(255)'
    
    def save_analysis(self, output_dir: str = "data/schema_analysis"):
        """Save all analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual schemas
        for filename, schema in self.schemas.items():
            schema_file = output_path / f"schema_{Path(filename).stem}.json"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, default=str)
        
        # Save comparison
        if self.comparisons:
            comparison_file = output_path / "schema_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(self.comparisons, f, indent=2, default=str)
        
        # Save unified schema
        unified = self.generate_unified_schema()
        unified_file = output_path / "unified_data_model.json"
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(unified, f, indent=2, default=str)
        
        # Save SQL DDL
        ddl_file = output_path / "create_tables.sql"
        with open(ddl_file, 'w', encoding='utf-8') as f:
            f.write("-- Data Model DDL\n")
            f.write(f"-- Generated: {datetime.now()}\n")
            f.write(f"-- Files analyzed: {len(self.schemas)}\n")
            f.write(f"-- Model mode: {self.model_mode}\n\n")
            for ddl in unified['sql_ddl']:
                f.write(ddl + "\n\n")

        # Save DuckDB load script (raw, one table per file, never mixes schemas)
        duckdb_file = output_path / "duckdb_load_raw.sql"
        with open(duckdb_file, 'w', encoding='utf-8') as f:
            f.write("-- DuckDB raw ingestion script (1 table per file)\n")
            f.write(f"-- Generated: {datetime.now()}\n")
            f.write("-- This script avoids mixing schemas by keeping files separate.\n\n")
            f.write("CREATE SCHEMA IF NOT EXISTS raw;\n\n")
            for filename, schema in self.schemas.items():
                table = f"raw.raw_{self._sanitize_identifier(Path(filename).stem)}"
                csv_path = Path(schema['filepath']).resolve().as_posix()
                name_map: Dict[str, str] = schema.get('column_name_map', {})
                original_cols = list(schema['columns'].keys())
                select_parts: List[str] = []
                for orig in original_cols:
                    sanitized = name_map.get(orig) or self._sanitize_identifier(orig)
                    # DuckDB trims whitespace around header names; strip to match its behavior.
                    header_for_select = str(orig).strip() or str(orig)
                    # header may contain quotes; escape by doubling
                    header_escaped = header_for_select.replace('"', '""')
                    select_parts.append(f'    "{header_escaped}" AS {sanitized}')
                select_sql = ",\n".join(select_parts)
                f.write(f"-- Source: {filename}\n")
                f.write(f"CREATE OR REPLACE TABLE {table} AS\n")
                f.write("SELECT\n")
                f.write(select_sql)
                f.write("\nFROM read_csv_auto(\n")
                f.write(f"  '{csv_path}',\n")
                f.write("  header=true\n");
                f.write(");\n\n")
            f.write("-- Optional: show loaded tables\n")
            f.write("-- SHOW TABLES IN raw;\n")
        
        # Save markdown report
        self._generate_markdown_report(output_path / "SCHEMA_ANALYSIS_REPORT.md")
        
        print(f"\n💾 Analysis saved to: {output_path}")
        return output_path


    def _generate_markdown_report(self, output_file: Path):
        """Generate comprehensive markdown report"""
        # Explicit UTF-8 avoids Windows default cp1252 choking on emojis.
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Multi-File Schema Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Files Analyzed:** {len(self.schemas)}\n\n")
            
            f.write("---\n\n")
            f.write("## 📁 Files Summary\n\n")
            f.write("| File | Type | Rows | Columns | Domain |\n")
            f.write("|------|------|------|---------|--------|\n")
            for filename, schema in self.schemas.items():
                f.write(f"| {filename} | {schema['file_type']} | {schema['row_count']:,} | "
                       f"{schema['column_count']} | {schema['business_domain']} |\n")
            
            f.write("\n---\n\n")
            f.write("## 🔍 Schema Comparison\n\n")
            
            if self.comparisons:
                f.write(f"**Compatibility:** {self.comparisons['schema_compatibility']}\n\n")
                f.write(f"**Unification Strategy:** {self.comparisons['unification_strategy']}\n\n")
                
                if self.comparisons['common_columns']:
                    f.write("### Common Columns\n\n")
                    for col in self.comparisons['common_columns']:
                        f.write(f"- {col}\n")
                
                f.write("\n### Recommendations\n\n")
                for rec in self.comparisons['recommendations']:
                    f.write(f"- {rec}\n")
            
            f.write("\n---\n\n")
            f.write("## 📊 Detailed Column Analysis\n\n")
            
            for filename, schema in self.schemas.items():
                f.write(f"### {filename}\n\n")
                f.write("| Column | Type | Business Meaning | Nulls % | Unique % |\n")
                f.write("|--------|------|------------------|---------|----------|\n")
                for col_name, col_info in schema['columns'].items():
                    f.write(f"| {col_name} | {col_info['inferred_type']} | "
                           f"{col_info['business_meaning']} | {col_info['null_percentage']:.1f}% | "
                           f"{col_info['unique_percentage']:.1f}% |\n")
                f.write("\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Analyze and compare schemas across multiple CSV files")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional list of CSV files to analyze (relative paths are resolved from repo root)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(repo_root / "data" / "source"),
        help="Directory containing source CSVs (default: data/source from repo root)",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV discovery within --data-dir (default: *.csv)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under --data-dir",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "data" / "schema_analysis"),
        help="Where to write schema analysis artifacts (default: data/schema_analysis from repo root)",
    )
    parser.add_argument(
        "--model-mode",
        choices=["separate", "by_type"],
        default="separate",
        help="How to generate modeled tables: 'separate' (one table per file, default) or 'by_type' (union columns by inferred file_type)",
    )
    args = parser.parse_args()

    analyzer = SchemaAnalyzer(model_mode=args.model_mode)

    # Determine CSV files to analyze
    if args.paths:
        csv_files: List[Path] = []
        for p in args.paths:
            path = Path(p)
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            csv_files.append(path)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = (repo_root / data_dir).resolve()
        if not data_dir.exists():
            raise SystemExit(f"Data directory not found: {data_dir}")
        csv_files = list(data_dir.rglob(args.pattern) if args.recursive else data_dir.glob(args.pattern))
    
    print(f"\n{'='*80}")
    print(f"🔬 MULTI-FILE SCHEMA ANALYSIS")
    print(f"{'='*80}")
    print(f"Repo root: {repo_root}")
    print(f"CWD: {Path.cwd()}")
    print(f"Files to analyze: {len(csv_files)}\n")
    
    # Analyze each file
    for csv_file in csv_files:
        try:
            analyzer.analyze_file(str(csv_file))
        except Exception as e:
            print(f"  ❌ Error analyzing {csv_file}: {e}")
    
    # Compare schemas
    if len(analyzer.schemas) >= 2:
        analyzer.compare_schemas()
    
    # Save results
    output_dir = analyzer.save_analysis(args.output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")