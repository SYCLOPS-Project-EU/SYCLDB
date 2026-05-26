package com.eurecom.calcite;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.List;
import java.util.Set;

public class SycldbJoin extends Join implements SycldbRel {

    public SycldbJoin(RelOptCluster cluster,
                      RelTraitSet traitSet,
                      List<RelHint> hints,
                      RelNode left,
                      RelNode right,
                      RexNode condition,
                      Set<CorrelationId> variablesSet,
                      JoinRelType joinType) {
        super(cluster, traitSet, hints, left, right, condition, variablesSet, joinType);
    }

    @Override
    public Join copy(RelTraitSet traitSet, RexNode conditionExpr, RelNode left, RelNode right, JoinRelType joinType, boolean semiJoinDone) {
        return new SycldbJoin(getCluster(), traitSet, hints, left, right, condition, variablesSet, joinType);
    }

    @Override
    public void implement(Implementor implementor) {
        // TODO
    }

    @Override
    public @Nullable RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        RelOptCost cost = super.computeSelfCost(planner, mq).multiplyBy(0.01);
        if (isSemiJoin()) {
            return cost.multiplyBy(0.1);
        }
        return cost;
    }
}
