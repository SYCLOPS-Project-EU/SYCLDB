package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.logical.LogicalProject;

public class SycldbProjectRule extends ConverterRule {
    public static final SycldbProjectRule INSTANCE = Config.INSTANCE
            .withConversion(LogicalProject.class, Convention.NONE,
                    SycldbRel.SYCLDB, "SycldbProjectRule")
            .withRuleFactory(SycldbProjectRule::new)
            .toRule(SycldbProjectRule.class);

    public SycldbProjectRule(Config config) {
        super(config);
    }

    @Override
    public RelNode convert(RelNode rel) {
        final LogicalProject project = (LogicalProject) rel;
        final RelTraitSet traitSet = project.getTraitSet().replace(out);
        return new SycldbProject(
                project.getCluster(),
                traitSet,
                convert(project.getInput(), out),
                project.getProjects(),
                project.getRowType()
        );
    }


}
