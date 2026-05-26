package com.eurecom.calcite;

import org.apache.calcite.DataContext;
import org.apache.calcite.adapter.java.AbstractQueryableTable;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.QueryProvider;
import org.apache.calcite.linq4j.Queryable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.ProjectableFilterableTable;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.schema.impl.AbstractTableQueryable;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.Collections;
import java.util.List;

public class SycldbTable extends AbstractQueryableTable implements ProjectableFilterableTable {
    private final String tableName;
    private final RelDataType dataType;
    private final double rowCount;

    SycldbTable(String tableName, RelDataType dataType, double rowCount) {
        super(Object[].class);
        this.tableName = tableName;
        this.dataType = dataType;
        this.rowCount = rowCount;
    }

    @Override
    public String toString() {
        return "SycldbTable {" + tableName + "}";
    }

//    @Override
//    public RelNode toRel(RelOptTable.ToRelContext context, RelOptTable relOptTable) {
//        final RelOptCluster cluster = context.getCluster();
//        return new SycldbTableScan(
//                cluster,
//                cluster.traitSetOf(SycldbRel.SYCLDB),
//                ImmutableList.of(),
//                relOptTable,
//                this
//        );
//    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        return typeFactory.copyType(dataType);
    }

    public List<String> getColumnNames() {
        return this.dataType.getFieldNames();
    }

    public String getTableName() {
        return tableName;
    }

    @Override
    public <T> Queryable<T> asQueryable(QueryProvider queryProvider, SchemaPlus schema, String tableName) {
        return new SycldbQueryable<>(queryProvider, schema, this, tableName);
    }

    @Override
    public Enumerable<@Nullable Object[]> scan(DataContext root, List<RexNode> filters, int @Nullable [] projects) {
        return null;
    }

    @Override
    public Statistic getStatistic() {
        return Statistics.of(
                rowCount,
                Collections.singletonList(
                        org.apache.calcite.util.ImmutableBitSet.of(0) // first column is always primary key
                )
        );
    }


    public static class SycldbQueryable<T> extends AbstractTableQueryable<T> {

        protected SycldbQueryable(QueryProvider queryProvider, SchemaPlus schema, SycldbTable table, String tableName) {
            super(queryProvider, schema, table, tableName);
        }

        @Override
        public Enumerator<T> enumerator() {
            throw new UnsupportedOperationException("enumerator");
        }
    }
}
