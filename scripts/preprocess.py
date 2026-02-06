"""
Robust Data Preprocessing and Loading Pipeline
Handles: Cleaning, Deduplication, Normalization, Validation, Sanitization
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import json
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline for sales data"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processing_log = []
        self.quality_issues = []
        self.transformations_applied = []
        self.header_like_rows: List[Dict[str, Any]] = []
        
    def log_action(self, action: str, details: str = ""):
        """Log preprocessing actions"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.processing_log.append(f"[{timestamp}] {action}: {details}")
        print(f"  ✓ {action}")
    
    def log_issue(self, issue_type: str, details: str, severity: str = "WARNING"):
        """Log data quality issues"""
        self.quality_issues.append({
            'type': issue_type,
            'details': details,
            'severity': severity
        })
    
    # ========================================================================
    # STEP 1: RAW DATA INGESTION
    # ========================================================================
    
    def ingest_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Ingest CSV with robust error handling
        Tries multiple encodings if needed
        """
        print(f"\n{'='*80}")
        print("📂 STEP 1: RAW DATA INGESTION")
        print(f"{'='*80}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No such file: {file_path}\n"
                f"- cwd: {Path.cwd()}\n"
                f"- resolved: {path.resolve()}\n"
                "Tip: pass an absolute path, or run from the repo root."
            )

        encodings = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings:
            try:
                df = pd.read_csv(str(path), encoding=enc, low_memory=False)
                self.log_action("CSV Loaded", f"Encoding: {enc}, Rows: {len(df):,}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"    ✗ Failed with {enc}: {e}")
                continue
        
        raise ValueError(f"Could not read CSV with any encoding: {encodings}")
    
    # ========================================================================
    # STEP 2: DATA CLEANING
    # ========================================================================
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning:
        - Standardize column names
        - Remove empty rows/columns
        - Trim whitespace
        - Remove special characters
        - Handle encoding issues
        """
        print(f"\n{'='*80}")
        print("🧹 STEP 2: DATA CLEANING")
        print(f"{'='*80}")
        
        initial_shape = df.shape
        
        # 2.1: Clean column names
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        df.columns = df.columns.str.upper()  # Uppercase
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Spaces to underscore
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars
        self.log_action("Column names cleaned", f"{len(df.columns)} columns")

        # 2.1b: Drop common unnamed/empty columns created by CSV exports
        unnamed_cols = [c for c in df.columns if c.startswith("UNNAMED")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            self.log_action("Unnamed columns dropped", f"{len(unnamed_cols)} columns")
        
        # 2.2: Remove completely empty rows
        df_before = len(df)
        df = df.dropna(how='all')
        rows_removed = df_before - len(df)
        if rows_removed > 0:
            self.log_action("Empty rows removed", f"{rows_removed} rows")
            self.log_issue("Empty Rows", f"Removed {rows_removed} completely empty rows", "INFO")
        
        # 2.3: Remove completely empty columns
        cols_before = len(df.columns)
        df = df.dropna(axis=1, how='all')
        cols_removed = cols_before - len(df.columns)
        if cols_removed > 0:
            self.log_action("Empty columns removed", f"{cols_removed} columns")
        
        # 2.4: Trim whitespace from all string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()
            # Remove multiple spaces
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        self.log_action("Whitespace trimmed", f"{len(string_columns)} string columns")
        
        # 2.5: Replace common null representations with actual NaN
        null_values = ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', '#N/A', 'nan']
        df = df.replace(null_values, np.nan)
        self.log_action("Null values standardized", "")
        
        # 2.6: Remove rows where index column has 'AVIN' or similar junk (from your data)
        if 'INDEX' in df.columns:
            junk_pattern = r'[A-Z]{2,}'  # Two or more consecutive letters
            if df['INDEX'].dtype == 'object':
                junk_rows = df['INDEX'].str.match(junk_pattern, na=False).sum()
                if junk_rows > 0:
                    df = df[~df['INDEX'].str.match(junk_pattern, na=False)]
                    self.log_action("Junk rows removed", f"{junk_rows} rows with invalid index")
                    self.log_issue("Data Quality", f"Found {junk_rows} rows with text in INDEX column", "WARNING")

        # 2.7: Extract header-like metadata rows (e.g., comparison tables with embedded headers)
        if len(df) > 0:
            def _is_header_like(row: pd.Series) -> bool:
                values = [str(v).strip().lower() for v in row.values]
                has_heads = any("heads" in v for v in values)
                has_price = any("price" in v for v in values)
                return has_heads and has_price

            header_like = df.apply(_is_header_like, axis=1)
            header_rows = int(header_like.sum())
            if header_rows > 0:
                self.header_like_rows = df[header_like].head(3).to_dict(orient="records")
                df = df[~header_like]
                self.log_action("Header-like rows extracted", f"{header_rows} rows")
        
        final_shape = df.shape
        self.log_action("Cleaning complete", 
                       f"Shape: {initial_shape} → {final_shape}")
        
        return df
    
    # ========================================================================
    # STEP 3: DEDUPLICATION
    # ========================================================================
    
    def deduplicate_data(self, df: pd.DataFrame, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate records
        - Exact duplicates
        - Near-duplicates (optional)
        """
        print(f"\n{'='*80}")
        print("🔍 STEP 3: DEDUPLICATION")
        print(f"{'='*80}")
        
        initial_rows = len(df)
        
        # 3.1: Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=subset, keep=keep)
        exact_dups = initial_rows - len(df_dedup)
        
        if exact_dups > 0:
            self.log_action("Exact duplicates removed", 
                           f"{exact_dups} rows ({exact_dups/initial_rows*100:.2f}%)")
            self.log_issue("Duplicates", f"Found {exact_dups} exact duplicate records", "WARNING")
        else:
            self.log_action("No exact duplicates found", "")
        
        # 3.2: Check for near-duplicates (same customer, date, amount)
        # This is domain-specific - customize as needed
        
        return df_dedup
    
    # ========================================================================
    # STEP 4: DATA TYPE DETECTION & NORMALIZATION
    # ========================================================================
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Normalize data types and formats:
        - Convert string numbers to numeric
        - Parse dates
        - Remove currency symbols
        - Standardize formats
        """
        print(f"\n{'='*80}")
        print("🔧 STEP 4: DATA NORMALIZATION")
        print(f"{'='*80}")
        
        column_mapping = {}
        
        # 4.1: Detect and map business columns
        key_columns = self._detect_business_columns(df)
        column_mapping = key_columns
        
        # 4.2: Normalize numeric columns (remove currency, convert to float)
        numeric_candidates = [key_columns.get('quantity'), 
                             key_columns.get('rate'), 
                             key_columns.get('amount')]
        
        for col in numeric_candidates:
            if col and col in df.columns:
                df[col] = self._normalize_numeric_column(df[col], col)

        # 4.2b: Normalize any other likely numeric amount columns not mapped
        for col in df.columns:
            col_lower = col.lower()
            if col in numeric_candidates:
                continue
            if any(tok in col_lower for tok in ['rate', 'price', 'amount', 'amt', 'gross', 'total']):
                df[col] = self._normalize_numeric_column(df[col], col)
        
        # 4.3: Parse and normalize dates
        if key_columns.get('date') and key_columns['date'] in df.columns:
            df['date_parsed'], date_format = self._normalize_date_column(
                df[key_columns['date']], key_columns['date']
            )
            
            # Create derived time columns
            if 'date_parsed' in df.columns:
                df['year'] = df['date_parsed'].dt.year
                df['quarter'] = df['date_parsed'].dt.quarter
                df['month'] = df['date_parsed'].dt.month
                df['month_name'] = df['date_parsed'].dt.strftime('%b-%y')
                df['day_of_week'] = df['date_parsed'].dt.dayofweek
                df['week_of_year'] = df['date_parsed'].dt.isocalendar().week
                self.log_action("Time dimensions created", "year, quarter, month, week")
        
        # 4.4: Normalize text columns (customer names, products)
        if key_columns.get('customer') and key_columns['customer'] in df.columns:
            df[key_columns['customer'] + '_normalized'] = self._normalize_text_column(
                df[key_columns['customer']], key_columns['customer']
            )
        
        # 4.5: Normalize categorical columns (sizes, etc.)
        if 'SIZE' in df.columns:
            df['size_normalized'] = df['SIZE'].str.upper().str.strip()
            # Map common variations
            size_mapping = {
                'SMALL': 'S', 'MEDIUM': 'M', 'LARGE': 'L',
                'XLARGE': 'XL', 'EXTRA LARGE': 'XL',
                '2XL': 'XXL', '3XL': 'XXXL'
            }
            df['size_normalized'] = df['size_normalized'].replace(size_mapping)
        
        return df, column_mapping
    
    def _detect_business_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect key business columns"""
        key_columns = {
            'date': None, 'month': None, 'customer': None,
            'product': None, 'sku': None, 'quantity': None,
            'rate': None, 'amount': None, 'size': None
        }
        
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
            elif 'size' in col_lower:
                key_columns['size'] = col
        
        self.log_action("Business columns detected", 
                       f"{sum(1 for v in key_columns.values() if v)} columns mapped")
        return key_columns
    
    def _normalize_numeric_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Normalize numeric column: remove currency, commas, convert to float"""
        if series.dtype == 'object':
            # Remove currency symbols
            cleaned = series.astype(str).str.replace(r'[₹$€£,]', '', regex=True)
            # Remove spaces
            cleaned = cleaned.str.replace(r'\s+', '', regex=True)
            # Convert to numeric
            converted = pd.to_numeric(cleaned, errors='coerce')
            
            nulls_introduced = converted.isna().sum() - series.isna().sum()
            if nulls_introduced > 0:
                self.log_issue("Numeric Conversion", 
                             f"{col_name}: {nulls_introduced} values couldn't be converted",
                             "WARNING")
            
            self.log_action(f"Normalized {col_name}", "Converted to numeric")
            return converted
        return series
    
    def _normalize_date_column(self, series: pd.Series, col_name: str) -> Tuple[pd.Series, str]:
        """Parse dates with multiple format attempts"""
        formats = [
            '%m-%d-%y',   # 06-05-21
            '%d-%m-%y',   # 05-06-21
            '%Y-%m-%d',   # 2021-06-05
            '%m/%d/%Y',   # 06/05/2021
            '%d/%m/%Y',   # 05/06/2021
            '%m-%d-%Y',   # 06-05-2021
            '%d-%m-%Y',   # 05-06-2021
        ]

        candidates: list[Tuple[str, pd.Series, float, int, int]] = []
        for fmt in formats:
            try:
                parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                success_rate = parsed.notna().sum() / len(series)
                if success_rate >= 0.9:  # 90% success
                    span_days = 0
                    distinct_months = 0
                    if parsed.notna().any():
                        span_days = int((parsed.max() - parsed.min()).days)
                        distinct_months = parsed.dt.to_period('M').nunique()
                    candidates.append((fmt, parsed, success_rate, distinct_months, span_days))
            except Exception:
                continue

        # Add auto-detect with dayfirst=True as a candidate for ambiguous data
        try:
            parsed_auto = pd.to_datetime(series, errors='coerce', dayfirst=True)
            success_rate = parsed_auto.notna().sum() / len(series)
            if success_rate >= 0.9:
                span_days = int((parsed_auto.max() - parsed_auto.min()).days)
                distinct_months = parsed_auto.dt.to_period('M').nunique()
                candidates.append(("auto_dayfirst", parsed_auto, success_rate, distinct_months, span_days))
        except Exception:
            pass

        if candidates:
            # Prefer higher success, then more distinct months, then wider span
            candidates.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
            fmt, parsed, success_rate, distinct_months, span_days = candidates[0]
            extra = f"Format: {fmt}, Success: {success_rate*100:.1f}%, Months: {distinct_months}, SpanDays: {span_days}"
            self.log_action(f"Date parsed ({col_name})", extra)
            return parsed, fmt

        # Fallback: auto-detection (no dayfirst bias)
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            success_rate = parsed.notna().sum() / len(series)
            self.log_action(f"Date parsed ({col_name})", f"Auto-detect, Success: {success_rate*100:.1f}%")
            return parsed, 'auto'
        except Exception:
            self.log_issue("Date Parsing", f"{col_name}: Could not parse dates", "ERROR")
            return series, None
    
    def _normalize_text_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Normalize text: title case, remove extra spaces"""
        if series.dtype == 'object':
            # Title case for names
            normalized = series.str.strip().str.title()
            # Remove multiple spaces
            normalized = normalized.str.replace(r'\s+', ' ', regex=True)
            self.log_action(f"Text normalized ({col_name})", "Title case applied")
            return normalized
        return series
    
    # ========================================================================
    # STEP 5: DATA VALIDATION
    # ========================================================================
    
    def validate_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Validate data quality and business logic:
        - Check for negative values
        - Validate relationships (amount = qty × rate)
        - Flag outliers
        - Check date ranges
        """
        print(f"\n{'='*80}")
        print("✅ STEP 5: DATA VALIDATION")
        print(f"{'='*80}")
        
        validation_flags = []
        
        # 5.1: Check for negative values in numeric columns
        qty_col = column_mapping.get('quantity')
        rate_col = column_mapping.get('rate')
        amt_col = column_mapping.get('amount')
        
        if qty_col and qty_col in df.columns:
            qty_numeric = pd.to_numeric(df[qty_col], errors='coerce')
            negative_qty = (qty_numeric < 0).sum()
            if negative_qty > 0:
                self.log_issue("Negative Values", f"{qty_col}: {negative_qty} negative quantities", "ERROR")
                df['flag_negative_qty'] = qty_numeric < 0
                validation_flags.append('flag_negative_qty')
        
        if rate_col and rate_col in df.columns:
            rate_numeric = pd.to_numeric(df[rate_col], errors='coerce')
            negative_rate = (rate_numeric < 0).sum()
            if negative_rate > 0:
                self.log_issue("Negative Values", f"{rate_col}: {negative_rate} negative rates", "ERROR")
                df['flag_negative_rate'] = rate_numeric < 0
                validation_flags.append('flag_negative_rate')
        
        # 5.2: Validate business logic: amount ≈ quantity × rate
        if all(col in df.columns for col in [qty_col, rate_col, amt_col]):
            qty_numeric = pd.to_numeric(df[qty_col], errors='coerce')
            rate_numeric = pd.to_numeric(df[rate_col], errors='coerce')
            amt_numeric = pd.to_numeric(df[amt_col], errors='coerce')
            
            calculated_amt = qty_numeric * rate_numeric
            tolerance = 1.0  # Allow ₹1 difference
            mismatch = abs(calculated_amt - amt_numeric) > tolerance
            mismatch_count = mismatch.sum()
            
            if mismatch_count > 0:
                self.log_issue("Business Logic", 
                             f"{mismatch_count} rows where amount ≠ qty × rate (±{tolerance})",
                             "WARNING")
                df['flag_amount_mismatch'] = mismatch
                validation_flags.append('flag_amount_mismatch')
        
        # 5.3: Check for outliers using IQR method
        if amt_col and amt_col in df.columns:
            amt_numeric = pd.to_numeric(df[amt_col], errors='coerce')
            Q1 = amt_numeric.quantile(0.25)
            Q3 = amt_numeric.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((amt_numeric < Q1 - 3*IQR) | (amt_numeric > Q3 + 3*IQR))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                self.log_issue("Outliers", 
                             f"{outlier_count} extreme outliers in {amt_col}",
                             "INFO")
                df['flag_amount_outlier'] = outliers
                validation_flags.append('flag_amount_outlier')
        
        # 5.4: Validate date ranges
        if 'date_parsed' in df.columns:
            min_date = df['date_parsed'].min()
            max_date = df['date_parsed'].max()
            
            # Check for future dates
            future_dates = (df['date_parsed'] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                self.log_issue("Date Range", f"{future_dates} future dates found", "WARNING")
                df['flag_future_date'] = df['date_parsed'] > pd.Timestamp.now()
                validation_flags.append('flag_future_date')
            
            # Check for very old dates (more than 10 years ago)
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=10)
            old_dates = (df['date_parsed'] < cutoff_date).sum()
            if old_dates > 0:
                self.log_issue("Date Range", f"{old_dates} dates older than 10 years", "INFO")

            # Check for implausible future years and null them out
            max_year = pd.Timestamp.now().year + 1
            invalid_future_years = df['date_parsed'].dt.year > max_year
            invalid_count = int(invalid_future_years.sum())
            if invalid_count > 0:
                self.log_issue("Date Range", f"{invalid_count} dates beyond year {max_year}", "WARNING")
                df.loc[invalid_future_years, 'date_parsed'] = pd.NaT
                df['flag_invalid_date'] = invalid_future_years
                validation_flags.append('flag_invalid_date')

            # Recompute derived time columns after date sanitization
            df['year'] = df['date_parsed'].dt.year
            df['quarter'] = df['date_parsed'].dt.quarter
            df['month'] = df['date_parsed'].dt.month
            df['month_name'] = df['date_parsed'].dt.strftime('%b-%y')
            df['day_of_week'] = df['date_parsed'].dt.dayofweek
            df['week_of_year'] = df['date_parsed'].dt.isocalendar().week
        
        # 5.5: Check for suspicious customer names (month names, etc.)
        customer_col = column_mapping.get('customer')
        if customer_col and customer_col in df.columns:
            month_pattern = r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}$'
            month_names = df[customer_col].str.match(month_pattern, na=False).sum()
            if month_names > 0:
                self.log_issue("Data Quality", 
                             f"{month_names} customer entries are month names",
                             "ERROR")
                df['flag_invalid_customer'] = df[customer_col].str.match(month_pattern, na=False)
                validation_flags.append('flag_invalid_customer')
        
        if validation_flags:
            self.log_action("Validation flags created", f"{len(validation_flags)} flag columns")
            df['has_quality_issues'] = df[validation_flags].any(axis=1)
        else:
            self.log_action("All validations passed", "No issues found")
        
        return df
    
    # ========================================================================
    # STEP 6: DATA SANITIZATION
    # ========================================================================
    
    def sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize data for security:
        - Remove SQL injection patterns
        - Remove HTML/script tags
        - Validate allowed characters
        """
        print(f"\n{'='*80}")
        print("🔒 STEP 6: DATA SANITIZATION")
        print(f"{'='*80}")
        
        string_columns = df.select_dtypes(include=['object']).columns
        
        # 6.1: Remove potential SQL injection patterns
        sql_patterns = [
            r"(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b)",  # SQL keywords
            r"(--|#|/\*|\*/)",  # SQL comments
            r"('|\")",  # Quotes (in non-standard positions)
        ]
        
        for col in string_columns:
            original = df[col].copy()
            for pattern in sql_patterns:
                # Just flag, don't modify data (might be legitimate)
                matches = df[col].astype(str).str.contains(pattern, case=False, regex=True, na=False).sum()
                if matches > 0:
                    self.log_issue("Security", 
                                 f"{col}: {matches} potential SQL patterns detected",
                                 "WARNING")
        
        # 6.2: Remove HTML/Script tags (if any)
        html_pattern = r'<[^>]+>'
        for col in string_columns:
            has_html = df[col].astype(str).str.contains(html_pattern, regex=True, na=False).sum()
            if has_html > 0:
                df[col] = df[col].astype(str).str.replace(html_pattern, '', regex=True)
                self.log_action(f"HTML removed from {col}", f"{has_html} entries cleaned")
        
        # 6.3: Remove null bytes and control characters
        for col in string_columns:
            df[col] = df[col].astype(str).str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
        
        self.log_action("Sanitization complete", f"{len(string_columns)} columns sanitized")
        
        return df
    
    # ========================================================================
    # STEP 7: FINAL CLEANUP & SUMMARY
    # ========================================================================
    
    def finalize_processing(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Final cleanup and generate processing summary
        """
        print(f"\n{'='*80}")
        print("📊 STEP 7: FINALIZATION & SUMMARY")
        print(f"{'='*80}")
        
        # 7.1: Drop rows with critical quality issues (optional)
        if 'flag_invalid_customer' in df.columns:
            invalid_customers = df['flag_invalid_customer'].sum()
            if invalid_customers > 0:
                print(f"\n⚠️  Found {invalid_customers} rows with invalid customers (month names)")
                print(f"   Recommendation: Filter these out before analysis")
        
        # 7.2: Create summary statistics
        summary = {
            'total_rows_processed': len(df),
            'columns': len(df.columns),
            'date_range': {
                'start': str(df['date_parsed'].min()) if 'date_parsed' in df.columns else None,
                'end': str(df['date_parsed'].max()) if 'date_parsed' in df.columns else None
            },
            'header_like_rows': self.header_like_rows,
            'missing_values': df.isnull().sum().to_dict(),
            'quality_issues': len(self.quality_issues),
            'validation_flags': [col for col in df.columns if col.startswith('flag_')],
            'column_mapping': column_mapping,
            'processing_log': self.processing_log,
            'quality_issues_list': self.quality_issues
        }
        
        # 7.3: Calculate business metrics
        amt_col = column_mapping.get('amount')
        if amt_col and amt_col in df.columns:
            amt_numeric = pd.to_numeric(df[amt_col], errors='coerce')
            summary['business_metrics'] = {
                'total_sales': float(amt_numeric.sum()),
                'avg_transaction': float(amt_numeric.mean()),
                'median_transaction': float(amt_numeric.median()),
                'max_transaction': float(amt_numeric.max()),
                'min_transaction': float(amt_numeric.min())
            }
        
        customer_col = column_mapping.get('customer')
        if customer_col and customer_col in df.columns:
            # Filter out month names first
            valid_customers = df[~df.get('flag_invalid_customer', False)] if 'flag_invalid_customer' in df.columns else df
            summary['unique_customers'] = int(valid_customers[customer_col].nunique())
        
        self.log_action("Summary generated", "")
        
        return df, summary
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def process(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full preprocessing pipeline
        """
        print(f"\n{'='*80}")
        print("🚀 DATA PREPROCESSING PIPELINE")
        print(f"{'='*80}")
        print(f"File: {file_path}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Ingest
        df = self.ingest_csv(file_path)
        
        # Step 2: Clean
        df = self.clean_data(df)
        
        # Step 3: Deduplicate
        df = self.deduplicate_data(df)
        
        # Step 4: Normalize
        df, column_mapping = self.normalize_data(df)
        
        # Step 5: Validate
        df = self.validate_data(df, column_mapping)
        
        # Step 6: Sanitize
        df = self.sanitize_data(df)
        
        # Step 7: Finalize
        df, summary = self.finalize_processing(df, column_mapping)
        
        print(f"\n{'='*80}")
        print("✅ PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total Processing Time: {len(self.processing_log)} steps")
        print(f"Quality Issues Found: {len(self.quality_issues)}")
        print(f"Final Dataset Shape: {df.shape}")
        
        return df, summary


def save_processed_data(
    df: pd.DataFrame,
    summary: Dict,
    output_dir: str = "data/processed",
    output_basename: str | None = None,
):
    """Save processed data and summary"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base = (output_basename or "processed_sales").strip()
    if not base:
        base = "processed_sales"
    if not base.endswith("_processed"):
        base = f"{base}_processed"

    # Save processed CSV
    csv_path = output_path / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Processed data saved: {csv_path}")
    
    # Save summary as JSON
    summary_path = output_path / f"{base}_summary.json"
    # Convert non-serializable objects
    summary_clean = json.loads(json.dumps(summary, default=str))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_clean, f, indent=2)
    print(f"💾 Summary saved: {summary_path}")
    
    # Save quality issues report
    if summary.get('quality_issues_list'):
        issues_path = output_path / f"{base}_quality_issues.txt"
        with open(issues_path, 'w', encoding='utf-8') as f:
            f.write("DATA QUALITY ISSUES REPORT\n")
            f.write("="*80 + "\n\n")
            for issue in summary['quality_issues_list']:
                f.write(f"[{issue['severity']}] {issue['type']}\n")
                f.write(f"  {issue['details']}\n\n")
        print(f"💾 Quality report saved: {issues_path}")
    
    return csv_path, summary_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Preprocess a retail sales CSV")
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(repo_root / "data" / "source" / "amazon_sale_report.csv"),
        help="Path to input CSV (default: data/source/amazon_sale_report.csv from repo root)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        # Treat relative paths as relative to repo root for consistent behavior
        csv_path = (repo_root / csv_path).resolve()
    
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    df_processed, summary = preprocessor.process(str(csv_path))
    
    # Save results
    save_processed_data(df_processed, summary)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}")
    
    if 'business_metrics' in summary:
        metrics = summary['business_metrics']
        print(f"\n💰 Business Metrics:")
        print(f"   Total Sales: ₹{metrics['total_sales']:,.2f}")
        print(f"   Avg Transaction: ₹{metrics['avg_transaction']:,.2f}")
        print(f"   Median Transaction: ₹{metrics['median_transaction']:,.2f}")
    
    if 'unique_customers' in summary:
        print(f"\n👥 Unique Customers: {summary['unique_customers']:,}")
    
    if summary.get('date_range', {}).get('start'):
        print(f"\n📅 Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    print(f"\n⚠️  Quality Issues: {summary['quality_issues']}")
    print(f"🔍 Validation Flags: {len(summary['validation_flags'])}")
    
    print(f"\n{'='*80}")