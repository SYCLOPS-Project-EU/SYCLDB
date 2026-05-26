package com.eurecom.calcite;

import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.runtime.PairList;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

public interface SycldbRel extends RelNode {
    Convention SYCLDB = new Convention.Impl("SYCLDB", SycldbRel.class);

    void implement(Implementor implementor);

//    int convertPlan(TreeConverter converter);


    class TreeConverter {
        ArrayList<String> tables = new ArrayList<>();
        TreeMap<Integer, String> idxToColumn = new TreeMap<>();
        HashMap<String, Integer> columnToIdx = new HashMap<>();
        ArrayList<OpNode> ops = new ArrayList<>();
        int columnCounter = 0;
        int idCounter = 0;

        int addTableColumns(List<String> columns, String tableName) {
            tables.add(tableName);
            for (String columnName : columns) {
                idxToColumn.put(columnCounter, columnName);
                columnToIdx.put(columnName, columnCounter);
                columnCounter++;
            }
            return idCounter++;
        }

//        int addFilter() {
//            OpNode op = new OpNode();
//            op.id = idCounter++;
//        }


        static class OpNode {
            int id;
            OpType type;
            ArrayList<Integer> inputs;

            @Nullable
            ArrayList<OpCondition> conditions;

            enum OpType {
                //                TABLE_SCAN,
                FILTER,
                PROJECT,
                JOIN,
                AGGREGATE,
            }

            static class OpCondition {
                enum Comparison {EQ, NE, LT, LE, GT, GE}

                enum Logical {NONE, AND, OR}

                enum OperandType {INTEGER, COLUMN}

                Logical condition;

                OperandType firstOperandType;
                @Nullable
                Integer firstOperand;
                @Nullable
                String firstColumn;

                Comparison comparison;

                OperandType secondOperandType;
                @Nullable
                Integer secondOperand;
                @Nullable
                String secondColumn;
            }

            OpNode(int id, OpType type, ArrayList<Integer> inputs, ArrayList<OpCondition> conditions) {
                this.id = id;
                this.type = type;
                this.inputs = inputs;
                this.conditions = conditions;
            }
        }
    }

    class Implementor {
        final PairList<@Nullable String, String> list = PairList.of();
        final RexBuilder rexBuilder;

        @Nullable
        RelOptTable table;

        @Nullable
        SycldbTable sycldbTable;

        public Implementor(RexBuilder rexBuilder) {
            this.rexBuilder = rexBuilder;
        }

        public void add(@Nullable String findOp, String aggOp) {
            list.add(findOp, aggOp);
        }

        public void visitChild(int ordinal, RelNode input) {
            assert ordinal == 0;
            ((SycldbRel) input).implement(this);
        }
    }
}
