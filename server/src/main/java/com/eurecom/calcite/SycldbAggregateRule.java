package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.InvalidRelException;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.checkerframework.checker.nullness.qual.Nullable;

import static com.eurecom.calcite.SycldbRel.SYCLDB;


public class SycldbAggregateRule extends ConverterRule {
    public SycldbAggregateRule(Config config) {
        super(config);
    }

    @Override
    public @Nullable RelNode convert(RelNode rel) {
        final LogicalAggregate agg = (LogicalAggregate) rel;
        final RelTraitSet traitSet = agg.getTraitSet().replace(out);
        try {
            return new SycldbAggregate(
                    rel.getCluster(),
                    traitSet,
                    convert(agg.getInput(), traitSet.simplify()),
                    agg.getGroupSet(),
                    agg.getGroupSets(),
                    agg.getAggCallList()
            );
        } catch (InvalidRelException e) {
            return null;
        }
    }

    public static final SycldbAggregateRule INSTANCE = Config.INSTANCE
            .withConversion(
                    LogicalAggregate.class,
                    Convention.NONE,
                    SYCLDB,
                    "SycldbAggregateRule"
            )
            .withRuleFactory(SycldbAggregateRule::new)
            .toRule(SycldbAggregateRule.class);
}
