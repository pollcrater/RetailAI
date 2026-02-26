import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.tools.duckdb_tools import DuckDBRunner, dataframe_to_json_preview, load_raw_from_sql_file


class FakeDataFrame:
    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def head(self, max_rows):
        return FakeDataFrame(self._rows[:max_rows])

    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError("Only records orient supported in tests")
        return list(self._rows)

    def itertuples(self, index=False):
        for row in self._rows:
            yield SimpleNamespace(**row)


class DuckDbToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test.duckdb"
        self.runner = DuckDBRunner(db_path=self.db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _mock_connection(self):
        con = MagicMock()
        con.execute.return_value = con
        return con

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_execute_runs_sql(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        mock_connect.return_value = con

        result = self.runner.execute("CREATE TABLE t (id INTEGER);")

        self.assertTrue(result.ok)
        print("mock_connect calls:", mock_connect.mock_calls)
        print("execute call args:", con.execute.call_args)
        con.execute.assert_called_once()
        con.close.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_execute_handles_errors(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        con.execute.side_effect = RuntimeError("boom")
        mock_connect.return_value = con

        result = self.runner.execute("CREATE TABLE t (id INTEGER);")

        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)
        con.close.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_query_rejects_non_select(self, mock_connect: MagicMock) -> None:
        result = self.runner.query("UPDATE t SET id = 1")

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "Only SELECT queries are allowed")
        mock_connect.assert_not_called()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_query_select_with_max_rows(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        con.fetch_df.return_value = FakeDataFrame(
            [
                {"id": 1, "amount": 10.5},
                {"id": 2, "amount": 20.0},
                {"id": 3, "amount": 30.25},
            ]
        )
        mock_connect.return_value = con

        result = self.runner.query("SELECT * FROM sales ORDER BY id", max_rows=2)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.content), 2)
        con.execute.assert_called_once()
        con.close.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_query_allows_with_select(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        con.fetch_df.return_value = FakeDataFrame([{"n": 4}])
        mock_connect.return_value = con

        result = self.runner.query(
            "WITH s AS (SELECT * FROM sales) SELECT count(*) AS n FROM s"
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.content.to_dict()[0]["n"], 4)
        con.execute.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_list_tables_returns_schema_table(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        con.fetch_df.return_value = FakeDataFrame(
            [{"table_schema": "main", "table_name": "sales"}]
        )
        mock_connect.return_value = con

        result = self.runner.list_tables()

        self.assertTrue(result.ok)
        self.assertIn("main.sales", result.content)
        con.execute.assert_called_once()
        con.close.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_get_schema_markdown(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        con.fetchall.return_value = [
            ("main", "sales", "id", "INTEGER"),
            ("main", "sales", "amount", "DOUBLE"),
        ]
        mock_connect.return_value = con

        result = self.runner.get_schema_markdown(schemas=("main",))

        self.assertTrue(result.ok)
        self.assertIn("### main.sales", result.content)
        self.assertIn("| id |", result.content)
        self.assertIn("| amount |", result.content)
        con.execute.assert_called_once()
        con.close.assert_called_once()

    @patch("src.tools.duckdb_tools.duckdb.connect")
    def test_load_raw_from_sql_file(self, mock_connect: MagicMock) -> None:
        con = self._mock_connection()
        mock_connect.return_value = con

        sql_file = Path(self._tmpdir.name) / "init.sql"
        sql_file.write_text("CREATE TABLE items (id INTEGER);", encoding="utf-8")
        result = load_raw_from_sql_file(sql_file=sql_file, db_path=self.db_path)

        self.assertTrue(result.ok)
        con.execute.assert_called_once()
        con.close.assert_called_once()

    def test_dataframe_to_json_preview(self) -> None:
        df = FakeDataFrame([{"a": 1}, {"a": 2}, {"a": 3}])
        payload = dataframe_to_json_preview(df, max_rows=2)
        data = json.loads(payload)
        self.assertEqual(len(data["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
