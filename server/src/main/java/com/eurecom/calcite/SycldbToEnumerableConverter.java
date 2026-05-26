package com.eurecom.calcite;

import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.linq4j.tree.BlockBuilder;
import org.apache.calcite.plan.ConventionTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterImpl;

import java.util.List;

public final class SycldbToEnumerableConverter extends ConverterImpl implements EnumerableRel {
    public SycldbToEnumerableConverter(RelNode child) {
        super(
                child.getCluster(),
                ConventionTraitDef.INSTANCE,
                child.getCluster().traitSetOf(EnumerableConvention.INSTANCE),
                child
        );
    }

    @Override
    public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
        return new SycldbToEnumerableConverter(inputs.get(0));
    }

    @Override
    public Result implement(EnumerableRelImplementor implementor, Prefer pref) {
//        try {
        BlockBuilder codeBlock = new BlockBuilder();
//            DeclarationStatement fieldStmt = Expressions.declare(0, "fields", Expressions.new_(LinkedHashMap.class));
//            codeBlock.add(fieldStmt);
//            Method putMethod = Map.class.getMethod("put", Object.class, Object.class);
//            final Expressions table = Expressions.constant(((SycldbRel) input).implement().table);
        PhysType physType = PhysTypeImpl.of(implementor.getTypeFactory(), getRowType(),
                pref.prefer(JavaRowFormat.ARRAY));
        return implementor.result(physType, codeBlock.toBlock());
//        } catch (NoSuchMethodException e) {
//            throw new RuntimeException(e);
//        }

//        return null;
    }
}
