# write the agent functionalities:
# Planner (SqlPlannerAgent): turns the user question into a safe DuckDB SELECT query.
# Executor (SqlExecutorAgent): runs the SQL against DuckDB and returns the dataframe.
# Validator (ResultValidatorAgent): checks results and generates the final answer or feedback for retries.