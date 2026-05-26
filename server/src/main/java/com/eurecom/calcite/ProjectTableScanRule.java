package com.eurecom.calcite;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;

import java.util.ArrayList;
import java.util.List;

public class ProjectTableScanRule extends RelOptRule {
    public static final ProjectTableScanRule INSTANCE =
            new ProjectTableScanRule();

    private ProjectTableScanRule() {
        super(operand(Project.class,
                        operand(SycldbTableScan.class, none())),
                "ProjectTableScanRule");
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        final Project project = call.rel(0);
        final SycldbTableScan scan = call.rel(1);

        List<RexNode> projectExprs = project.getProjects();
        List<Integer> projectedFields = new ArrayList<>();

        // Ensure all expressions are simple RexInputRef
        for (RexNode rex : projectExprs) {
            if (rex instanceof RexInputRef) {
                projectedFields.add(((RexInputRef) rex).getIndex());
            } else {
                // Not a simple projection — bail out
                return;
            }
        }

        int[] newProjects = projectedFields.stream().mapToInt(i -> i).toArray();

        // Push projection into a new TableScan
        RelNode newScan = new SycldbTableScan(
                scan.getCluster(),
                scan.getTraitSet(),
                scan.getHints(),
                scan.getTable(),
                scan.sycldbTable,
                newProjects
        );

        call.transformTo(newScan);
    }
}

