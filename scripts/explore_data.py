"""
Data Exploration Script for Retail Sales CSV
Analyzes the structure, statistics, and patterns in the sales data
Handles multiple CSVs and data quality issues
"""

import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_column_names(df):
    """Standardize column names"""
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
    return df

def detect_date_format(date_series):
    """Detect and parse date formats"""
    # Try different date formats
    formats = [
        '%m-%d-%y',  # 06-05-21
        '%d-%m-%y',  # 05-06-21
        '%Y-%m-%d',  # 2021-06-05
        '%m/%d/%Y',  # 06/05/2021
        '%d/%m/%Y',  # 05/06/2021
    ]
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
            if parsed.notna().sum() > len(date_series) * 0.9:  # 90% success rate
                return parsed, fmt
        except:
            continue
    
    # Fallback to auto-detection
    try:
        return pd.to_datetime(date_series, errors='coerce'), 'auto'
    except:
        return date_series, None

def explore_csv(file_path: str):
    """Comprehensive data exploration with robust error handling"""
    
    print("=" * 80)
    print("🔍 RETAIL SALES DATA EXPLORATION REPORT")
    print("=" * 80)
    
    try:
        # Load CSV with error handling
        print(f"\n📂 Loading: {file_path}")
        
        # First, try to detect issues
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            print(f"Header: {first_line.strip()}")
        
        # Load with pandas
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        
        # Clean column names
        df = clean_column_names(df)
        
        print("✅ CSV loaded successfully!")
        print(f"Raw columns: {df.columns.tolist()}")
        
        # Data Quality Check - Remove completely empty rows
        initial_rows = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_rows:
            print(f"⚠️  Removed {initial_rows - len(df)} completely empty rows")
        
        # Basic Information
        print("\n" + "=" * 80)
        print("📊 BASIC INFORMATION")
        print("=" * 80)
        print(f"Total Rows: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Identify key business columns
        key_columns = {
            'date': None,
            'month': None,
            'customer': None,
            'product': None,
            'sku': None,
            'quantity': None,
            'rate': None,
            'amount': None,
            'region': None,
            'category': None
        }
        
        # Map columns to business concepts
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower and not key_columns['date']:
                key_columns['date'] = col
            elif 'month' in col_lower and not key_columns['month']:
                key_columns['month'] = col
            elif 'customer' in col_lower and not key_columns['customer']:
                key_columns['customer'] = col
            elif 'style' in col_lower and not key_columns['product']:
                key_columns['product'] = col
            elif 'sku' in col_lower and not key_columns['sku']:
                key_columns['sku'] = col
            elif 'pcs' in col_lower or 'qty' in col_lower or 'quantity' in col_lower:
                key_columns['quantity'] = col
            elif 'rate' in col_lower or 'price' in col_lower:
                key_columns['rate'] = col
            elif 'amt' in col_lower or 'amount' in col_lower or 'total' in col_lower or 'gross' in col_lower:
                key_columns['amount'] = col
            elif 'region' in col_lower:
                key_columns['region'] = col
            elif 'category' in col_lower or 'type' in col_lower:
                key_columns['category'] = col
        
        print("\n🎯 Identified Business Columns:")
        for key, col in key_columns.items():
            if col:
                print(f"  • {key.upper():12s} → {col}")
        
        # Column Information
        print("\n" + "=" * 80)
        print("📋 COLUMNS & DATA TYPES")
        print("=" * 80)
        for idx, (col, dtype) in enumerate(df.dtypes.items(), 1):
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            print(f"{idx:2d}. {col:25s} | {str(dtype):15s} | Nulls: {null_count:6d} ({null_pct:5.2f}%) | Unique: {unique_count:,}")
        
        # Parse dates if date column exists
        if key_columns['date']:
            print(f"\n📅 Parsing date column: {key_columns['date']}")
            df['parsed_date'], date_format = detect_date_format(df[key_columns['date']])
            if date_format:
                print(f"✅ Date format detected: {date_format}")
                df['year'] = df['parsed_date'].dt.year
                df['quarter'] = df['parsed_date'].dt.quarter
                df['month_num'] = df['parsed_date'].dt.month
                print(f"Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
            else:
                print(f"⚠️  Could not parse dates automatically")
        
        # Sample Data
        print("\n" + "=" * 80)
        print("🎯 SAMPLE DATA (First 5 Rows)")
        print("=" * 80)
        print(df.head().to_string())
        
        # Identify Column Types
        print("\n" + "=" * 80)
        print("🔑 COLUMN ANALYSIS")
        print("=" * 80)
        
        date_columns = []
        categorical_columns = []
        numeric_columns = []
        text_columns = []
        
        for col in df.columns:
            if col in ['parsed_date', 'year', 'quarter', 'month_num']:
                continue
                
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            non_null_count = df[col].notna().sum()
            
            # Skip if mostly null
            if non_null_count < len(df) * 0.1:
                col_type = "⚠️  MOSTLY_NULL"
            elif col == key_columns['date'] or 'date' in col.lower():
                date_columns.append(col)
                col_type = "📅 DATE"
            elif df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
                if col == key_columns['quantity'] or col == key_columns['rate'] or col == key_columns['amount']:
                    col_type = "💰 METRIC"
                else:
                    col_type = "🔢 NUMERIC"
            elif df[col].dtype == 'object':
                if unique_ratio < 0.05:  # Less than 5% unique = categorical
                    categorical_columns.append(col)
                    col_type = "🏷️  CATEGORY"
                elif unique_ratio > 0.95:  # More than 95% unique = ID/text
                    text_columns.append(col)
                    col_type = "🆔 ID/TEXT"
                else:
                    categorical_columns.append(col)
                    col_type = "🏷️  CATEGORY"
            else:
                col_type = "❓ OTHER"
            
            print(f"{col_type} | {col:25s} | Unique: {unique_count:8,} ({unique_ratio*100:5.2f}%)")
        
        # Statistical Summary for Numeric Columns
        if numeric_columns:
            print("\n" + "=" * 80)
            print("📈 NUMERIC COLUMNS STATISTICS")
            print("=" * 80)
            numeric_stats = df[numeric_columns].describe()
            print(numeric_stats.to_string())
            
            # Additional business metrics
            if key_columns['amount']:
                # Convert to numeric if it's stored as string
                amount_col = df[key_columns['amount']].copy()
                if amount_col.dtype == 'object':
                    amount_col = pd.to_numeric(amount_col, errors='coerce')
                total_amount = amount_col.sum()
                avg_amount = amount_col.mean()
                print(f"\n💰 Total Sales Amount: {total_amount:,.2f}")
                print(f"💰 Average Transaction: {avg_amount:,.2f}")
            
            if key_columns['quantity']:
                # Convert to numeric if it's stored as string
                qty_col = df[key_columns['quantity']].copy()
                if qty_col.dtype == 'object':
                    qty_col = pd.to_numeric(qty_col, errors='coerce')
                total_qty = qty_col.sum()
                print(f"📦 Total Units Sold: {total_qty:,.0f}")
        
        # Categorical Analysis
        if categorical_columns:
            print("\n" + "=" * 80)
            print("🏷️  CATEGORICAL COLUMNS ANALYSIS (Top 10 values)")
            print("=" * 80)
            for col in categorical_columns[:5]:  # Show first 5 categorical columns
                print(f"\n▶️  {col}:")
                value_counts = df[col].value_counts().head(10)
                for idx, (val, count) in enumerate(value_counts.items(), 1):
                    pct = (count / len(df)) * 100
                    print(f"  {idx:2d}. {str(val)[:40]:40s}: {count:6,} ({pct:5.2f}%)")
        
        # Time-based Analysis
        if 'parsed_date' in df.columns and df['parsed_date'].notna().sum() > 0:
            print("\n" + "=" * 80)
            print("📅 TIME-BASED ANALYSIS")
            print("=" * 80)
            
            if 'year' in df.columns:
                print("\n📊 Sales by Year:")
                if key_columns['amount']:
                    # Convert amount to numeric
                    df_temp = df.copy()
                    df_temp['amount_numeric'] = pd.to_numeric(df_temp[key_columns['amount']], errors='coerce')
                    yearly = df_temp.groupby('year')['amount_numeric'].agg(['sum', 'count', 'mean'])
                    yearly.columns = ['Total Amount', 'Transactions', 'Avg Amount']
                    print(yearly.to_string())
            
            if 'quarter' in df.columns and key_columns['amount']:
                print("\n📊 Sales by Quarter (Recent Year):")
                if 'year' in df.columns:
                    recent_year = df['year'].max()
                    recent_data = df[df['year'] == recent_year].copy()
                    recent_data['amount_numeric'] = pd.to_numeric(recent_data[key_columns['amount']], errors='coerce')
                    quarterly = recent_data.groupby('quarter')['amount_numeric'].agg(['sum', 'count'])
                    quarterly.columns = ['Total Amount', 'Transactions']
                    print(quarterly.to_string())
        
        # Data Quality Issues
        print("\n" + "=" * 80)
        print("⚠️  DATA QUALITY CHECKS")
        print("=" * 80)
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n🔴 Columns with Missing Values:")
            for col, count in missing[missing > 0].items():
                print(f"  • {col:25s}: {count:6,} ({count/len(df)*100:5.2f}%)")
        else:
            print("✅ No missing values found")
        
        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            print(f"\n⚠️  Duplicate Rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
        else:
            print("\n✅ No duplicate rows found")
        
        # Schema Summary for DuckDB
        print("\n" + "=" * 80)
        print("🗄️  RECOMMENDED DUCKDB SCHEMA")
        print("=" * 80)
        
        schema_mapping = {
            'int64': 'INTEGER',
            'float64': 'DOUBLE',
            'object': 'VARCHAR',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        print("CREATE TABLE sales (")
        columns_added = []
        for idx, (col, dtype) in enumerate(df.dtypes.items()):
            if col in ['parsed_date', 'year', 'quarter', 'month_num']:
                continue
            duckdb_type = schema_mapping.get(str(dtype), 'VARCHAR')
            if col in date_columns or 'date' in col.lower():
                duckdb_type = 'DATE'
            col_name = col.lower().replace(' ', '_').replace('-', '_')
            columns_added.append(col_name)
            comma = ","
            print(f"    {col_name:30s} {duckdb_type}{comma}")
        
        # Add derived columns
        if 'parsed_date' in df.columns:
            print(f"    {'year':30s} INTEGER,")
            print(f"    {'quarter':30s} INTEGER,")
            print(f"    {'month_num':30s} INTEGER")
        
        print(");")
        
        # Create normalized schema with key mappings
        print("\n" + "=" * 80)
        print("🔑 STANDARDIZED COLUMN MAPPING")
        print("=" * 80)
        
        column_mapping = {}
        for biz_col, actual_col in key_columns.items():
            if actual_col:
                normalized = actual_col.lower().replace(' ', '_').replace('-', '_')
                column_mapping[biz_col] = normalized
                print(f"{biz_col:15s} → {normalized}")
        
        # Save exploration results
        summary_path = Path("data/data_exploration_summary.txt")
        summary_path.parent.mkdir(exist_ok=True)
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA EXPLORATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Rows: {len(df):,}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Date Range: {df['parsed_date'].min()} to {df['parsed_date'].max()}\n\n" if 'parsed_date' in df.columns else "\n")
            
            f.write("Business Column Mapping:\n")
            for biz_col, actual_col in key_columns.items():
                if actual_col:
                    f.write(f"  {biz_col}: {actual_col}\n")
            
            f.write(f"\nDate Columns: {date_columns}\n")
            f.write(f"Categorical Columns: {categorical_columns}\n")
            f.write(f"Numeric Columns: {numeric_columns}\n")
            
            if key_columns['amount']:
                amount_col = pd.to_numeric(df[key_columns['amount']], errors='coerce')
                total_amount = amount_col.sum()
                f.write(f"\nTotal Sales Amount: {total_amount:,.2f}\n")
            
            if key_columns['quantity']:
                qty_col = pd.to_numeric(df[key_columns['quantity']], errors='coerce')
                total_qty = qty_col.sum()
                f.write(f"Total Units Sold: {total_qty:,.0f}\n")
        
        # Save column mapping as JSON for code use
        import json
        mapping_path = Path("data/column_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({
                'key_columns': column_mapping,
                'categorical': [c.lower().replace(' ', '_') for c in categorical_columns],
                'numeric': [c.lower().replace(' ', '_') for c in numeric_columns],
                'date': [c.lower().replace(' ', '_') for c in date_columns]
            }, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_path}")
        print(f"✅ Column mapping saved to: {mapping_path}")
        
        # Generate sample queries based on actual data
        print("\n" + "=" * 80)
        print("💡 SUGGESTED EXAMPLE QUERIES (Based on Your Data)")
        print("=" * 80)
        
        example_queries = []
        
        # Basic queries
        if key_columns['amount']:
            example_queries.append("What is the total sales amount?")
            example_queries.append("Show me the top 10 transactions by amount")
        
        # Time-based queries
        if 'parsed_date' in df.columns:
            example_queries.append("What were the total sales in 2021?")
            example_queries.append("Show me sales trend by month")
            example_queries.append("Compare Q1 vs Q2 sales")
            if 'year' in df.columns and df['year'].nunique() > 1:
                years = sorted(df['year'].unique())
                if len(years) >= 2:
                    example_queries.append(f"What is the year-over-year growth from {years[-2]} to {years[-1]}?")
        
        # Customer queries
        if key_columns['customer']:
            top_customer = df[key_columns['customer']].value_counts().index[0] if len(df) > 0 else None
            if top_customer:
                example_queries.append(f"Show me all purchases by {top_customer}")
            example_queries.append("Who are the top 5 customers by sales amount?")
            example_queries.append("How many unique customers do we have?")
        
        # Product queries
        if key_columns['product']:
            example_queries.append("Which products generated the most revenue?")
            example_queries.append("What is the average price per product style?")
        
        # Category/Region queries
        if categorical_columns:
            cat_col = categorical_columns[0]
            example_queries.append(f"Show me sales breakdown by {cat_col}")
            if len(categorical_columns) > 1:
                cat_col2 = categorical_columns[1]
                example_queries.append(f"Compare {cat_col} performance across different {cat_col2}")
        
        # Complex analytical queries
        if key_columns['amount'] and 'parsed_date' in df.columns:
            example_queries.append("What is the monthly average order value trend?")
            example_queries.append("Which month had the highest sales?")
        
        for i, query in enumerate(example_queries, 1):
            print(f"{i:2d}. {query}")
        
        # Save example queries
        queries_path = Path("data/example_queries.txt")
        with open(queries_path, 'w') as f:
            for query in example_queries:
                f.write(f"{query}\n")
        print(f"\n✅ Example queries saved to: {queries_path}")
        
        print("\n" + "=" * 80)
        print("✅ DATA EXPLORATION COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Summary: {len(df):,} rows analyzed")
        print(f"📅 Time period: {df['parsed_date'].min()} to {df['parsed_date'].max()}" if 'parsed_date' in df.columns else "")
        if key_columns['amount']:
            amount_col = pd.to_numeric(df[key_columns['amount']], errors='coerce')
            print(f"💰 Total sales: {amount_col.sum():,.2f}")
        print(f"👥 Unique customers: {df[key_columns['customer']].nunique():,}" if key_columns['customer'] else "")
        
        return {
            'df': df,
            'key_columns': key_columns,
            'date_columns': date_columns,
            'categorical_columns': categorical_columns,
            'numeric_columns': numeric_columns,
            'column_mapping': column_mapping
        }
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        print("Please place your sales CSV in the data/ folder")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during exploration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    csv_path = "data/source/sample_sales.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"Starting exploration of: {csv_path}\n")
    result = explore_csv(csv_path)