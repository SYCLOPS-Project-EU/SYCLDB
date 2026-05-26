package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalJoin;

import static com.eurecom.calcite.SycldbRel.SYCLDB;


public class SycldbJoinRule extends ConverterRule {
    public SycldbJoinRule(Config config) {
        super(config);
    }

    @Override
    public RelNode convert(RelNode rel) {
        final LogicalJoin input = (LogicalJoin) rel;

        return new SycldbJoin(
                rel.getCluster(),
                input.getTraitSet().replace(out),
                input.getHints(),
                convert(input.getLeft(), out),
                convert(input.getRight(), out),
                input.getCondition(),
                input.getVariablesSet(),
                input.getJoinType()
        );
    }

    public static final SycldbJoinRule INSTANCE = Config.INSTANCE
            .withConversion(
                    LogicalJoin.class,
                    Convention.NONE,
                    SYCLDB,
                    "SycldbJoinRule"
            )
            .withRuleFactory(SycldbJoinRule::new)
            .toRule(SycldbJoinRule.class);
}
