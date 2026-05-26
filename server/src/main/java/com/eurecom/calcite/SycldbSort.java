package com.eurecom.calcite;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.Collections;
import java.util.List;

import static java.util.Objects.requireNonNull;

public class SycldbSort extends Sort implements SycldbRel {

    public SycldbSort(RelOptCluster cluster,
                      RelTraitSet traits,
                      List<RelHint> hints,
                      RelNode child,
                      RelCollation collation,
                      @Nullable RexNode offset,
                      @Nullable RexNode fetch) {
        super(cluster, traits, hints, child, collation, offset, fetch);
        assert getConvention() == SycldbRel.SYCLDB;
        assert getConvention() == child.getConvention();
    }

    @Override
    public void implement(Implementor implementor) {
        // TODO
    }

    @Override
    public Sort copy(RelTraitSet traitSet, RelNode newInput, RelCollation newCollation, @Nullable RexNode offset, @Nullable RexNode fetch) {
        return new SycldbSort(getCluster(), traitSet, Collections.emptyList(), newInput, newCollation, offset, fetch);
    }

    @Override
    public @Nullable RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        final RelOptCost cost = requireNonNull(super.computeSelfCost(planner, mq));
        return cost.multiplyBy(0.1);
    }
}
