package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalFilter;

public class SycldbFilterRule extends ConverterRule {
    public static final SycldbFilterRule INSTANCE = Config.INSTANCE
            .withConversion(LogicalFilter.class, Convention.NONE,
                    SycldbRel.SYCLDB, "SycldbFilterRule")
            .withRuleFactory(SycldbFilterRule::new)
            .toRule(SycldbFilterRule.class);

    public SycldbFilterRule(Config config) {
        super(config);
    }

    @Override
    public RelNode convert(RelNode relNode) {
        final LogicalFilter filter = (LogicalFilter) relNode;
        final RelTraitSet traitSet = filter.getTraitSet().replace(out);
        return new SycldbFilter(
                relNode.getCluster(),
                traitSet,
                convert(filter.getInput(), out),
                filter.getCondition()
        );
    }
}
