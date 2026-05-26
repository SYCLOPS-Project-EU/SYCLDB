package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalSort;

public class SycldbSortRule extends ConverterRule {
    public static final SycldbSortRule INSTANCE = Config.INSTANCE
            .withConversion(LogicalSort.class, Convention.NONE,
                    SycldbRel.SYCLDB, "SycldbSortRule")
            .withRuleFactory(SycldbSortRule::new)
            .toRule(SycldbSortRule.class);

    public SycldbSortRule(Config config) {
        super(config);
    }

    @Override
    public RelNode convert(RelNode rel) {
        final LogicalSort sort = (LogicalSort) rel;
        return new SycldbSort(
                sort.getCluster(),
                sort.getTraitSet().replace(out),
                sort.getHints(),
                convert(sort.getInput(), out),
                sort.getCollation(),
                sort.offset, sort.fetch
        );
    }
}
