package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalTableScan;

import java.util.Collections;

import static com.eurecom.calcite.SycldbRel.SYCLDB;

public final class SycldbTableScanRule extends ConverterRule {
    public SycldbTableScanRule(final Config config) {
        super(config);
    }

    @Override
    public RelNode convert(final RelNode rel) {
        final LogicalTableScan scan = (LogicalTableScan) rel;
        final SycldbTable table = scan.getTable().unwrap(SycldbTable.class);
        if (table != null) {
            return new SycldbTableScan(
                    scan.getCluster(),
                    scan.getCluster().traitSetOf(SYCLDB),
                    Collections.emptyList(),
                    scan.getTable(),
                    table,
                    null
            );
        }
        return null;
    }


    public static final SycldbTableScanRule INSTANCE = Config.INSTANCE
            .withConversion(
                    LogicalTableScan.class,
                    Convention.NONE,
                    SYCLDB,
                    "SycldbTableScanRule"
            )
            .withRuleFactory(SycldbTableScanRule::new)
            .toRule(SycldbTableScanRule.class);
}
