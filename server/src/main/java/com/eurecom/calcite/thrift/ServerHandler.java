package com.eurecom.calcite.thrift;

// generated code

import com.eurecom.calcite.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlExplainFormat;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.*;
import org.apache.thrift.TException;
import org.apache.thrift.server.TServer;

import java.util.Collections;

public class ServerHandler implements CalciteServer.Iface {
    private final FrameworkConfig config;
    private final Program program;
    private final SycldbSchema schema;

    public ServerHandler(TServer server) {
        super();
        SchemaPlus rootSchema = Frameworks.createRootSchema(true);
        schema = new SycldbSchema(SycldbSchema.SchemaOptions.SSB);
        SchemaPlus sycldbSchema = rootSchema.add("SYCLDBSCHEMA", schema);

        SqlParser.Config parserConfig = SqlParser.config()
                .withCaseSensitive(false); // parser: not case sensitive

        SqlValidator.Config validatorConfig = SqlValidator.Config.DEFAULT
                .withIdentifierExpansion(true); // expand * to full columns

        SqlToRelConverter.Config converterConfig = SqlToRelConverter.config()
                .withDecorrelationEnabled(true)
                .withTrimUnusedFields(true);

        this.program = Programs.ofRules(
                // SYCLDB rules
                SycldbFilterRule.INSTANCE,
                SycldbProjectRule.INSTANCE,
                SycldbToEnumerableConverterRule.INSTANCE,
                SycldbTableScanRule.INSTANCE,
                SycldbJoinRule.INSTANCE,
                SycldbAggregateRule.INSTANCE,
                SycldbSortRule.INSTANCE,

                // optimization rules

                //        TODO: doesn't work
//                ProjectTableScanRule.INSTANCE,

                CoreRules.FILTER_INTO_JOIN,
                CoreRules.AGGREGATE_MERGE,
                CoreRules.AGGREGATE_PROJECT_PULL_UP_CONSTANTS,
                CoreRules.AGGREGATE_PROJECT_MERGE,
                CoreRules.FILTER_MERGE,
                CoreRules.FILTER_AGGREGATE_TRANSPOSE,
                CoreRules.FILTER_SCAN,
                CoreRules.PROJECT_AGGREGATE_MERGE,
                CoreRules.PROJECT_MERGE,
                CoreRules.PROJECT_REMOVE,
                CoreRules.PROJECT_TO_SEMI_JOIN,
                CoreRules.PROJECT_JOIN_TRANSPOSE,
                CoreRules.JOIN_CONDITION_PUSH,
                CoreRules.JOIN_ADD_REDUNDANT_SEMI_JOIN,
                CoreRules.JOIN_ON_UNIQUE_TO_SEMI_JOIN,
                CoreRules.JOIN_TO_SEMI_JOIN,
                CoreRules.FILTER_SUB_QUERY_TO_CORRELATE,
                CoreRules.PROJECT_SUB_QUERY_TO_CORRELATE,
                CoreRules.JOIN_SUB_QUERY_TO_CORRELATE
        );


//        planner.addRule(CoreRules.AGGREGATE_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_FILTER_TRANSPOSE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_JOIN_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_REMOVE);
//        planner.addRule(CoreRules.AGGREGATE_JOIN_TRANSPOSE);

//        planner.addRule(CoreRules.PROJECT_JOIN_JOIN_REMOVE);
//        planner.addRule(CoreRules.PROJECT_JOIN_REMOVE);

//        planner.addRule(CoreRules.JOIN_COMMUTE);
//        planner.addRule(CoreRules.JOIN_PUSH_EXPRESSIONS);
//        planner.addRule(CoreRules.JOIN_PUSH_TRANSITIVE_PREDICATES);

        this.config = Frameworks.newConfigBuilder()
                .defaultSchema(sycldbSchema)
                .parserConfig(parserConfig)
                .sqlValidatorConfig(validatorConfig)
                .sqlToRelConverterConfig(converterConfig)
                .programs(program)
                .build();
    }

    @Override
    public void ping() throws TException {
        System.out.println("ping()");
    }

    @Override
    public void shutdown() throws TException {
        System.out.println("shutdown()");
//        server.stop();
    }

    private static RelOptCluster newCluster(RelDataTypeFactory factory) {
        RelOptPlanner planner = new VolcanoPlanner();
        planner.addRelTraitDef(ConventionTraitDef.INSTANCE);
        return RelOptCluster.create(planner, new RexBuilder(factory));
    }

    private static final RelOptTable.ViewExpander NOOP_EXPANDER = (rowType, queryString, schemaPath
            , viewPath) -> null;

    @Override
    public PlanResult parse(String sql) {
        long start = System.nanoTime();


        // need to trim the sql string as it seems it is not trimed prior to here
        sql = sql.trim();
        // remove last charcter if it is a ;
        if (sql.length() > 0 && sql.charAt(sql.length() - 1) == ';') {
            sql = sql.substring(0, sql.length() - 1);
        }

        Planner planner = Frameworks.getPlanner(config);

        SqlNode parsed = null;
        try {
            parsed = planner.parse(sql);
        } catch (SqlParseException e) {
            System.err.println("Sql ParseException: " + e.getMessage());
            throw new RuntimeException(e);
        }

        System.out.println("Parsed");

        SqlNode validated = null;
        try {
            validated = planner.validate(parsed);
        } catch (ValidationException e) {
            System.err.println("ValidationException: " + e.getMessage());
            throw new RuntimeException(e);
        }

        System.out.println("Validated");

        RelRoot root = null;
        try {
            root = planner.rel(validated);
        } catch (RelConversionException e) {
            System.err.println("RelConversionException: " + e.getMessage());
            throw new RuntimeException(e);
        }

        System.out.println(
                RelOptUtil.dumpPlan("[Logical plan]", root.rel, SqlExplainFormat.TEXT,
                        SqlExplainLevel.NON_COST_ATTRIBUTES));


        RelTraitSet traitSet = root.rel.getTraitSet()
                .replace(SycldbRel.SYCLDB);
        // Start the optimization process to obtain the most efficient physical plan based on the
        // provided rule set.
        RelOptPlanner relOptPlanner = root.rel.getCluster().getPlanner();
        RelNode physical = program.run(relOptPlanner, root.rel, traitSet,
                Collections.emptyList(), Collections.emptyList());


        String json = RelOptUtil.dumpPlan("", physical, SqlExplainFormat.JSON, SqlExplainLevel.NO_ATTRIBUTES);

        System.out.println(
                RelOptUtil.dumpPlan("[Physical plan]", physical, SqlExplainFormat.TEXT,
                        SqlExplainLevel.NON_COST_ATTRIBUTES));

        System.out.println(json);
        SycldbJsonConverter converter = new SycldbJsonConverter(json);


        long end = System.nanoTime();

        System.out.println((end - start) / 1000);

        return new PlanResult(converter.getRels(), json);
    }
}
