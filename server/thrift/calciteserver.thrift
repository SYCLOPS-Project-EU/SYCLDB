namespace java com.eurecom.calcite.thrift

enum RelNodeType {
    TABLE_SCAN,
    FILTER,
    JOIN,
    PROJECT,
    AGGREGATE,
    SORT,
}

enum DirectionOption {
    DESCENDING,
    ASCENDING,
}

enum NullsOption {
    FIRST,
    LAST,
}

struct CollationType {
    1: i64 field,
    2: DirectionOption direction,
    3: NullsOption nulls,
}

struct AggType {
    1: string agg,
    2: list<i64> operands,
    3: string name,
    4: string type,
    5: bool distinct,
}

enum ExprOption {
    LITERAL,
    COLUMN,
    EXPR,
}

enum LiteralOption {
    LITERAL,
    RANGE,
}

struct LiteralType {
    1: LiteralOption literalOption,
    2: optional i64 value,                      // literal
    3: optional list<list<string>> rangeSet,    // range
}

struct ExprType {
    1: ExprOption exprType,
    2: optional i64 input,                  // column
    3: optional string name,                // column
    4: optional string op,                  // expr
    5: optional list<ExprType> operands     // expr
    6: optional LiteralType literal,        // literal
    7: optional string type,                // literal, expr
}

struct RelNode {
    1: i64 id,
    2: RelNodeType relOp,
    3: optional list<string> tables,            // table scan
    4: optional list<i64> inputs,               // table scan, join
    5: optional ExprType condition,             // filter, join
    6: optional string joinType,                // join
    7: optional list<string> fields,            // project
    8: optional list<ExprType> exprs,           // project
    9: optional list<i64> group,                // aggregate
    10: optional list<AggType> aggs,            // aggregate
    11: optional list<CollationType> collation, // sort
}

struct PlanResult {
    1: list<RelNode> rels,
    2: string oldJson,
}

service CalciteServer {
    void ping()
    void shutdown()
    PlanResult parse(1: string sql)
}