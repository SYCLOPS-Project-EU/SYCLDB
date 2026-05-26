package com.eurecom.calcite;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.checkerframework.checker.nullness.qual.Nullable;

import static java.util.Objects.requireNonNull;

public class SycldbFilter extends Filter implements SycldbRel {

    public SycldbFilter(RelOptCluster cluster, RelTraitSet traitSet, RelNode child, RexNode condition) {
        super(cluster, traitSet, child, condition);
        assert getConvention() == SycldbRel.SYCLDB;
        assert getConvention() == child.getConvention();
    }

    @Override
    public void implement(Implementor implementor) {
        //TODO
    }

//    @Override
//    public int convertPlan(TreeConverter converter) {
//        return 0;
//    }

    @Override
    public Filter copy(RelTraitSet traitSet, RelNode input, RexNode condition) {
        return new SycldbFilter(getCluster(), traitSet, input, condition);
    }

    @Override
    public @Nullable RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        final RelOptCost cost = requireNonNull(super.computeSelfCost(planner, mq));
        return cost.multiplyBy(0.1);
    }
}
