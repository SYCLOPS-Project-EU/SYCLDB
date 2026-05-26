package com.eurecom.calcite;

import com.google.common.collect.ImmutableList;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.InvalidRelException;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.ImmutableBitSet;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static java.util.Objects.requireNonNull;

public class SycldbAggregate extends Aggregate implements SycldbRel {

    private static final Set<SqlKind> SUPPORTED_AGGREGATIONS =
            EnumSet.of(SqlKind.COUNT, SqlKind.SUM, SqlKind.SUM0, SqlKind.AVG);

    public SycldbAggregate(RelOptCluster cluster,
                           RelTraitSet traitSet,
                           RelNode input,
                           ImmutableBitSet groupSet,
                           List<ImmutableBitSet> groupSets,
                           List<AggregateCall> aggCalls) throws InvalidRelException {
        super(cluster, traitSet, ImmutableList.of(), input, groupSet, groupSets, aggCalls);

        assert getConvention() == input.getConvention();
        assert getConvention() == SycldbRel.SYCLDB;

        for (AggregateCall aggCall : aggCalls) {
            final SqlKind kind = aggCall.getAggregation().getKind();
            if (!SUPPORTED_AGGREGATIONS.contains(kind)) {
                final String message =
                        String.format(Locale.ROOT,
                                "Aggregation %s not supported (use one of %s)", kind,
                                SUPPORTED_AGGREGATIONS);
                throw new InvalidRelException(message);
            }
        }

        if (getGroupType() != Group.SIMPLE) {
            final String message = String.format(Locale.ROOT, "Only %s grouping is supported. "
                    + "Yours is %s", Group.SIMPLE, getGroupType());
            throw new InvalidRelException(message);
        }
    }

    @Override
    public Aggregate copy(RelTraitSet traitSet, RelNode input, ImmutableBitSet groupSet, @Nullable List<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls) {
        try {
            return new SycldbAggregate(getCluster(), traitSet, input,
                    groupSet, groupSets, aggCalls);
        } catch (InvalidRelException e) {
            throw new AssertionError(e);
        }
    }

    @Override
    public void implement(Implementor implementor) {
        // todo
    }

    @Override
    public @Nullable RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
        return requireNonNull(super.computeSelfCost(planner, mq)).multiplyBy(0.1);
    }
}
