//package com.eurecom.calcite;
//
//import org.apache.calcite.DataContext;
//import org.apache.calcite.adapter.java.JavaTypeFactory;
//import org.apache.calcite.config.CalciteConnectionConfig;
//import org.apache.calcite.config.CalciteConnectionConfigImpl;
//import org.apache.calcite.config.CalciteConnectionProperty;
//import org.apache.calcite.jdbc.CalciteSchema;
//import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
//import org.apache.calcite.linq4j.Enumerable;
//import org.apache.calcite.linq4j.Linq4j;
//import org.apache.calcite.linq4j.QueryProvider;
//import org.apache.calcite.plan.*;
//import org.apache.calcite.plan.volcano.VolcanoPlanner;
//import org.apache.calcite.prepare.CalciteCatalogReader;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.rules.CoreRules;
//import org.apache.calcite.rel.type.RelDataType;
//import org.apache.calcite.rel.type.RelDataTypeFactory;
//import org.apache.calcite.rex.RexBuilder;
//import org.apache.calcite.schema.ScannableTable;
//import org.apache.calcite.schema.SchemaPlus;
//import org.apache.calcite.schema.impl.AbstractTable;
//import org.apache.calcite.sql.SqlExplainFormat;
//import org.apache.calcite.sql.SqlExplainLevel;
//import org.apache.calcite.sql.SqlNode;
//import org.apache.calcite.sql.fun.SqlStdOperatorTable;
//import org.apache.calcite.sql.parser.SqlParser;
//import org.apache.calcite.sql.type.SqlTypeName;
//import org.apache.calcite.sql.validate.SqlValidator;
//import org.apache.calcite.sql.validate.SqlValidatorUtil;
//import org.apache.calcite.sql2rel.SqlToRelConverter;
//import org.apache.calcite.sql2rel.StandardConvertletTable;
//
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.Collections;
//import java.util.List;
//import java.util.Properties;
//
//public class SYCLDBQueryProcessor {
//
//    public static void main(String[] args) throws Exception {
//        if (args.length != 1) {
//            System.err.println("SQL_FILE parameter missing or too many parameters");
//            System.exit(-1);
//        }
//
//        String sqlQuery = Files.readString(Paths.get(args[0]));
//
//        // Instantiate a type factory for creating types (e.g., VARCHAR, NUMERIC, etc.)
//        RelDataTypeFactory typeFactory = new JavaTypeFactoryImpl();
//        // Create the root schema describing the data model
//        CalciteSchema schema = CalciteSchema.createRootSchema(false);
//
//        RelDataTypeFactory.Builder lineorderType = new RelDataTypeFactory.Builder(typeFactory);
//        lineorderType.add("lo_orderkey", SqlTypeName.INTEGER);
//        lineorderType.add("lo_linenumber", SqlTypeName.INTEGER);
//        lineorderType.add("lo_custkey", SqlTypeName.INTEGER);
//        lineorderType.add("lo_partkey", SqlTypeName.INTEGER);
//        lineorderType.add("lo_suppkey", SqlTypeName.INTEGER);
//        lineorderType.add("lo_orderdate", SqlTypeName.INTEGER);
//        lineorderType.add("lo_orderpriority", SqlTypeName.VARCHAR);
//        lineorderType.add("lo_shippriority", SqlTypeName.VARCHAR);
//        lineorderType.add("lo_quantity", SqlTypeName.INTEGER);
//        lineorderType.add("lo_extendedprice", SqlTypeName.FLOAT);
//        lineorderType.add("lo_ordtotalprice", SqlTypeName.FLOAT);
//        lineorderType.add("lo_discount", SqlTypeName.FLOAT);
//        lineorderType.add("lo_revenue", SqlTypeName.FLOAT);
//        lineorderType.add("lo_supplycost", SqlTypeName.FLOAT);
//        lineorderType.add("lo_tax", SqlTypeName.INTEGER);
//        lineorderType.add("lo_commitdate", SqlTypeName.INTEGER);
//        lineorderType.add("lo_shopmode", SqlTypeName.VARCHAR);
//
//        SycldbTable lineorderTable = new SycldbTable("lineorder", lineorderType.build());
//        schema.add("lineorder", lineorderTable);
//
//
//        RelDataTypeFactory.Builder partType = new RelDataTypeFactory.Builder(typeFactory);
//        partType.add("p_partkey", SqlTypeName.INTEGER);
//        partType.add("p_name", SqlTypeName.VARCHAR);
//        partType.add("p_mfgr", SqlTypeName.VARCHAR);
//        partType.add("p_category", SqlTypeName.VARCHAR);
//        partType.add("p_brand1", SqlTypeName.VARCHAR);
//        partType.add("p_color", SqlTypeName.VARCHAR);
//        partType.add("p_type", SqlTypeName.VARCHAR);
//        partType.add("p_size", SqlTypeName.INTEGER);
//        partType.add("p_container", SqlTypeName.VARCHAR);
//
//        SycldbTable partTable = new SycldbTable("part", partType.build());
//        schema.add("part", partTable);
//
//        RelDataTypeFactory.Builder supplierType = new RelDataTypeFactory.Builder(typeFactory);
//        supplierType.add("s_suppkey", SqlTypeName.INTEGER);
//        supplierType.add("s_name", SqlTypeName.VARCHAR);
//        supplierType.add("s_address", SqlTypeName.VARCHAR);
//        supplierType.add("s_city", SqlTypeName.VARCHAR);
//        supplierType.add("s_nation", SqlTypeName.VARCHAR);
//        supplierType.add("s_region", SqlTypeName.VARCHAR);
//        supplierType.add("s_phone", SqlTypeName.VARCHAR);
//
//        SycldbTable supplierTable = new SycldbTable("supplier", supplierType.build());
//        schema.add("supplier", supplierTable);
//
//        RelDataTypeFactory.Builder customerType = new RelDataTypeFactory.Builder(typeFactory);
//        customerType.add("c_custkey", SqlTypeName.INTEGER);
//        customerType.add("c_name", SqlTypeName.VARCHAR);
//        customerType.add("c_address", SqlTypeName.VARCHAR);
//        customerType.add("c_city", SqlTypeName.VARCHAR);
//        customerType.add("c_nation", SqlTypeName.VARCHAR);
//        customerType.add("c_region", SqlTypeName.VARCHAR);
//        customerType.add("c_phone", SqlTypeName.VARCHAR);
//        customerType.add("c_mktsegment", SqlTypeName.VARCHAR);
//
//        SycldbTable customerTable = new SycldbTable("customer", customerType.build());
//        schema.add("customer", customerTable);
//
//        RelDataTypeFactory.Builder ddateType = new RelDataTypeFactory.Builder(typeFactory);
//        ddateType.add("d_datekey", SqlTypeName.INTEGER);
//        ddateType.add("d_date", SqlTypeName.VARCHAR);
//        ddateType.add("d_dayofweek", SqlTypeName.VARCHAR);
//        ddateType.add("d_month", SqlTypeName.VARCHAR);
//        ddateType.add("d_year", SqlTypeName.INTEGER);
//        ddateType.add("d_yearmonthnum", SqlTypeName.INTEGER);
//        ddateType.add("d_yearmonth", SqlTypeName.VARCHAR);
//        ddateType.add("d_daynuminweek", SqlTypeName.INTEGER);
//        ddateType.add("d_daynuminmonth", SqlTypeName.INTEGER);
//        ddateType.add("d_daynuminyear", SqlTypeName.INTEGER);
//        ddateType.add("d_monthnuminyear", SqlTypeName.INTEGER);
//        ddateType.add("d_weeknuminyear", SqlTypeName.INTEGER);
//        ddateType.add("d_sellingseasin", SqlTypeName.VARCHAR);
//        ddateType.add("d_lastdayinweekfl", SqlTypeName.INTEGER);
//        ddateType.add("d_lastdayinmonthfl", SqlTypeName.INTEGER);
//        ddateType.add("d_holidayfl", SqlTypeName.INTEGER);
//        ddateType.add("d_weekdayfl", SqlTypeName.INTEGER);
//
//        SycldbTable ddateTable = new SycldbTable("ddate", ddateType.build());
//        schema.add("ddate", ddateTable);
//
//        // Create an SQL parser
//        SqlParser parser = SqlParser.create(sqlQuery);
//        // Parse the query into an AST
//        SqlNode sqlNode = parser.parseQuery();
//
//        // Configure and instantiate validator
//        Properties props = new Properties();
//        props.setProperty(CalciteConnectionProperty.CASE_SENSITIVE.camelName(), "false");
//        CalciteConnectionConfig config = new CalciteConnectionConfigImpl(props);
//        CalciteCatalogReader catalogReader = new CalciteCatalogReader(schema,
//                Collections.singletonList(""),
//                typeFactory, config);
//
//        SqlValidator validator = SqlValidatorUtil.newValidator(SqlStdOperatorTable.instance(),
//                catalogReader, typeFactory,
//                SqlValidator.Config.DEFAULT);
//
//        // Validate the initial AST
//        SqlNode validNode = validator.validate(sqlNode);
//
//        // Configure and instantiate the converter of the AST to Logical plan (requires opt cluster)
//        RelOptCluster cluster = newCluster(typeFactory);
//        SqlToRelConverter relConverter = new SqlToRelConverter(
//                NOOP_EXPANDER,
//                validator,
//                catalogReader,
//                cluster,
//                StandardConvertletTable.INSTANCE,
//                SqlToRelConverter.config());
//
//        // Convert the valid AST into a logical plan
//        RelNode logPlan = relConverter.convertQuery(validNode, false, true).rel;
//
//        // Display the logical plan
//        System.out.println(
//                RelOptUtil.dumpPlan("[Logical plan]", logPlan, SqlExplainFormat.TEXT,
//                        SqlExplainLevel.EXPPLAN_ATTRIBUTES));
/// /        System.out.println(
/// /                RelOptUtil.dumpPlan(
/// /                        "[Logical plan 2]",
/// /                        logPlan,
/// /                        SqlExplainFormat.JSON,
/// /                        SqlExplainLevel.NO_ATTRIBUTES
/// /                )
/// /        );
//
//        RelOptPlanner planner = cluster.getPlanner();
//
//        // Initialize optimizer/planner with the necessary rules
//        planner.addRule(SycldbFilterRule.INSTANCE);
//        planner.addRule(SycldbProjectRule.INSTANCE);
//        planner.addRule(SycldbToEnumerableConverterRule.INSTANCE);
//        planner.addRule(SycldbTableScanRule.INSTANCE);
//        planner.addRule(SycldbJoinRule.INSTANCE);
//        planner.addRule(SycldbAggregateRule.INSTANCE);
//
//        planner.addRule(CoreRules.FILTER_INTO_JOIN);
//        planner.addRule(CoreRules.AGGREGATE_MERGE);
//        planner.addRule(CoreRules.AGGREGATE_PROJECT_PULL_UP_CONSTANTS);
//        planner.addRule(CoreRules.AGGREGATE_PROJECT_MERGE);
//        planner.addRule(CoreRules.AGGREGATE_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_FILTER_TRANSPOSE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_JOIN_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_TRANSPOSE_EXTENDED);
//        planner.addRule(CoreRules.FILTER_MERGE);
//        planner.addRule(CoreRules.FILTER_AGGREGATE_TRANSPOSE);
//        planner.addRule(CoreRules.PROJECT_JOIN_JOIN_REMOVE);
//        planner.addRule(CoreRules.PROJECT_JOIN_REMOVE);
//        planner.addRule(CoreRules.PROJECT_MERGE);
//        planner.addRule(CoreRules.PROJECT_REMOVE);
//        planner.addRule(CoreRules.JOIN_CONDITION_PUSH);
//        planner.addRule(CoreRules.JOIN_COMMUTE);
//        planner.addRule(CoreRules.JOIN_PUSH_EXPRESSIONS);
//        planner.addRule(CoreRules.JOIN_PUSH_TRANSITIVE_PREDICATES);
//
////        planner.addRule(EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE);
////        planner.addRule(CoreRules.PROJECT_TO_CALC);
////        planner.addRule(CoreRules.FILTER_TO_CALC);
////        planner.addRule(EnumerableRules.ENUMERABLE_CALC_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_JOIN_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_SORT_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_LIMIT_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_AGGREGATE_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_VALUES_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_UNION_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_MINUS_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_INTERSECT_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_MATCH_RULE);
////        planner.addRule(EnumerableRules.ENUMERABLE_WINDOW_RULE);
//
//
//        // Define the type of the output plan (in this case we want a physical plan in
//        // BindableConvention)
//        logPlan = planner.changeTraits(logPlan,
//                cluster.traitSet().replace(SycldbRel.SYCLDB));
//        planner.setRoot(logPlan);
//        // Start the optimization process to obtain the most efficient physical plan based on the
//        // provided rule set.
//        RelNode phyPlan = planner.findBestExp();
//
//        // Display the physical plan
//        System.out.println(
//                RelOptUtil.dumpPlan("[Physical plan]", phyPlan, SqlExplainFormat.TEXT,
//                        SqlExplainLevel.NON_COST_ATTRIBUTES));
//
//        // get executable plan
////        Bindable<Object[]> executablePlan = EnumerableInterpretable.toBindable(
////                new HashMap<>(),
////                null,
////                phyPlan,
////                EnumerableRel.Prefer.ARRAY
////        );
////
////        // Run the executable plan using a context simply providing access to the schema
////        for (Object[] row : executablePlan.bind(new SchemaOnlyDataContext(schema))) {
////            System.out.println(Arrays.toString(row));
////        }
//    }
//
//    /**
//     * A simple table based on a list.
//     */
//    private static class ListTable extends AbstractTable implements ScannableTable {
//        private final RelDataType rowType;
//        private final List<Object[]> data;
//
//        ListTable(RelDataType rowType, List<Object[]> data) {
//            this.rowType = rowType;
//            this.data = data;
//        }
//
//        @Override
//        public Enumerable<Object[]> scan(final DataContext root) {
//            return Linq4j.asEnumerable(data);
//        }
//
//        @Override
//        public RelDataType getRowType(final RelDataTypeFactory typeFactory) {
//            return rowType;
//        }
//    }
//
//    private static RelOptCluster newCluster(RelDataTypeFactory factory) {
//        RelOptPlanner planner = new VolcanoPlanner();
//        planner.addRelTraitDef(ConventionTraitDef.INSTANCE);
//        return RelOptCluster.create(planner, new RexBuilder(factory));
//    }
//
//    private static final RelOptTable.ViewExpander NOOP_EXPANDER = (rowType, queryString, schemaPath
//            , viewPath) -> null;
//
//    /**
//     * A simple data context only with schema information.
//     */
//    private static final class SchemaOnlyDataContext implements DataContext {
//        private final SchemaPlus schema;
//
//        SchemaOnlyDataContext(CalciteSchema calciteSchema) {
//            this.schema = calciteSchema.plus();
//        }
//
//        @Override
//        public SchemaPlus getRootSchema() {
//            return schema;
//        }
//
//        @Override
//        public JavaTypeFactory getTypeFactory() {
//            return new JavaTypeFactoryImpl();
//        }
//
//        @Override
//        public QueryProvider getQueryProvider() {
//            return null;
//        }
//
//        @Override
//        public Object get(final String name) {
//            return null;
//        }
//    }
//}