package com.eurecom.calcite;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;

import static com.eurecom.calcite.SycldbRel.SYCLDB;

public class SycldbToEnumerableConverterRule extends ConverterRule {
    public SycldbToEnumerableConverterRule(Config config) {
        super(config);
    }

    @Override
    public RelNode convert(RelNode rel) {
        return new SycldbToEnumerableConverter(rel);
    }

    public static final SycldbToEnumerableConverterRule INSTANCE = Config.INSTANCE
            .withConversion(
                    SycldbRel.class,
                    SYCLDB,
                    EnumerableConvention.INSTANCE,
                    "SycldbToEnumerableConverterRule"
            )
            .withRuleFactory(SycldbToEnumerableConverterRule::new)
            .toRule(SycldbToEnumerableConverterRule.class);
}
