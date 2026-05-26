package com.eurecom.calcite;

import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.impl.AbstractSchema;
import org.apache.calcite.sql.type.SqlTypeName;

import java.util.HashMap;
import java.util.Map;

public class SycldbSchema extends AbstractSchema {
    private final Map<String, Table> tables;
    private final RelDataTypeFactory typeFactory;
    private final CalciteSchema rootSchema;

    public enum SchemaOptions {
        SSB,
        TPCH,
    }

    public SycldbSchema() {
        this(SchemaOptions.SSB);
    }

    public SycldbSchema(SchemaOptions options) {
        tables = new HashMap<String, Table>();
        typeFactory = new JavaTypeFactoryImpl();

        switch (options) {
            case TPCH: {
                System.out.println("Loading TPCH schema");

                RelDataTypeFactory.Builder supplierType = typeFactory.builder()
                        .add("s_suppkey", SqlTypeName.INTEGER)
                        .add("s_name", SqlTypeName.CHAR)
                        .add("s_address", SqlTypeName.VARCHAR)
                        .add("s_nationkey", SqlTypeName.INTEGER)
                        .add("s_phone", SqlTypeName.CHAR)
                        .add("s_acctbal", SqlTypeName.DECIMAL)
                        .add("s_comment", SqlTypeName.VARCHAR);

                SycldbTable supplierTable = new SycldbTable("supplier", supplierType.build(), 100000);
                tables.put("supplier", supplierTable);

                RelDataTypeFactory.Builder partType = typeFactory.builder()
                        .add("p_partkey", SqlTypeName.INTEGER)
                        .add("p_name", SqlTypeName.VARCHAR)
                        .add("p_mfgr", SqlTypeName.CHAR)
                        .add("p_brand", SqlTypeName.CHAR)
                        .add("p_type", SqlTypeName.VARCHAR)
                        .add("p_size", SqlTypeName.INTEGER)
                        .add("p_container", SqlTypeName.CHAR)
                        .add("p_retailprice", SqlTypeName.DECIMAL)
                        .add("p_comment", SqlTypeName.VARCHAR);

                SycldbTable partTable = new SycldbTable("part", partType.build(), 2000000);
                tables.put("part", partTable);

                RelDataTypeFactory.Builder partsuppType = typeFactory.builder()
                        .add("ps_partkey", SqlTypeName.INTEGER)
                        .add("ps_suppkey", SqlTypeName.INTEGER)
                        .add("ps_availqty", SqlTypeName.INTEGER)
                        .add("ps_supplycost", SqlTypeName.DECIMAL)
                        .add("ps_comment", SqlTypeName.VARCHAR);

                SycldbTable partsuppTable = new SycldbTable("partsupp", partsuppType.build(), 8000000);
                tables.put("partsupp", partsuppTable);

                RelDataTypeFactory.Builder lineitemType = typeFactory.builder()
                        .add("l_orderkey", SqlTypeName.INTEGER)
                        .add("l_partkey", SqlTypeName.INTEGER)
                        .add("l_suppkey", SqlTypeName.INTEGER)
                        .add("l_linenumber", SqlTypeName.INTEGER)
                        .add("l_quantity", SqlTypeName.DECIMAL)
                        .add("l_extendedprice", SqlTypeName.DECIMAL)
                        .add("l_discount", SqlTypeName.DECIMAL)
                        .add("l_tax", SqlTypeName.DECIMAL)
                        .add("l_returnflag", SqlTypeName.CHAR)
                        .add("l_linestatus", SqlTypeName.CHAR)
                        .add("l_shipdate", SqlTypeName.DATE)
                        .add("l_commitdate", SqlTypeName.DATE)
                        .add("l_receiptdate", SqlTypeName.DATE)
                        .add("l_shipinstruct", SqlTypeName.CHAR)
                        .add("l_shipmode", SqlTypeName.CHAR)
                        .add("l_comment", SqlTypeName.VARCHAR);

                SycldbTable lineitemTable = new SycldbTable("lineitem", lineitemType.build(), 59986052);
                tables.put("lineitem", lineitemTable);

                RelDataTypeFactory.Builder ordersType = typeFactory.builder()
                        .add("o_orderkey", SqlTypeName.INTEGER)
                        .add("o_custkey", SqlTypeName.INTEGER)
                        .add("o_orderstatus", SqlTypeName.CHAR)
                        .add("o_totalprice", SqlTypeName.DECIMAL)
                        .add("o_orderdate", SqlTypeName.DATE)
                        .add("o_orderpriority", SqlTypeName.CHAR)
                        .add("o_clerk", SqlTypeName.CHAR)
                        .add("o_shippriority", SqlTypeName.INTEGER)
                        .add("o_comment", SqlTypeName.VARCHAR);

                SycldbTable ordersTable = new SycldbTable("orders", ordersType.build(), 15000000);
                tables.put("orders", ordersTable);

                RelDataTypeFactory.Builder customerType = typeFactory.builder()
                        .add("c_custkey", SqlTypeName.INTEGER)
                        .add("c_name", SqlTypeName.VARCHAR)
                        .add("c_address", SqlTypeName.VARCHAR)
                        .add("c_nationkey", SqlTypeName.INTEGER)
                        .add("c_phone", SqlTypeName.CHAR)
                        .add("c_acctbal", SqlTypeName.DECIMAL)
                        .add("c_mktsegment", SqlTypeName.CHAR)
                        .add("c_comment", SqlTypeName.VARCHAR);

                SycldbTable customerTable = new SycldbTable("customer", customerType.build(), 1500000);
                tables.put("customer", customerTable);

                RelDataTypeFactory.Builder nationType = typeFactory.builder()
                        .add("n_nationkey", SqlTypeName.INTEGER)
                        .add("n_name", SqlTypeName.CHAR)
                        .add("n_regionkey", SqlTypeName.INTEGER)
                        .add("n_comment", SqlTypeName.VARCHAR);

                SycldbTable nationTable = new SycldbTable("nation", nationType.build(), 25);
                tables.put("nation", nationTable);

                RelDataTypeFactory.Builder regionType = typeFactory.builder()
                        .add("r_regionkey", SqlTypeName.INTEGER)
                        .add("r_name", SqlTypeName.CHAR)
                        .add("r_comment", SqlTypeName.VARCHAR);

                SycldbTable regionTable = new SycldbTable("region", regionType.build(), 5);
                tables.put("region", regionTable);

                break;
            }
            case SSB:
            default: {
                System.out.println("Loading SSB schema");

                RelDataTypeFactory.Builder lineorderType = new RelDataTypeFactory.Builder(typeFactory)
                        .add("lo_orderkey", SqlTypeName.INTEGER)
                        .add("lo_linenumber", SqlTypeName.INTEGER)
                        .add("lo_custkey", SqlTypeName.INTEGER)
                        .add("lo_partkey", SqlTypeName.INTEGER)
                        .add("lo_suppkey", SqlTypeName.INTEGER)
                        .add("lo_orderdate", SqlTypeName.INTEGER)
                        .add("lo_orderpriority", SqlTypeName.VARCHAR)
                        .add("lo_shippriority", SqlTypeName.VARCHAR)
                        .add("lo_quantity", SqlTypeName.INTEGER);

//        lineorderType.add("lo_extendedprice", SqlTypeName.FLOAT);
//        lineorderType.add("lo_ordtotalprice", SqlTypeName.FLOAT);
//        lineorderType.add("lo_discount", SqlTypeName.FLOAT);
//        lineorderType.add("lo_revenue", SqlTypeName.FLOAT);
//        lineorderType.add("lo_supplycost", SqlTypeName.FLOAT);
                lineorderType.add("lo_extendedprice", SqlTypeName.INTEGER);
                lineorderType.add("lo_ordtotalprice", SqlTypeName.INTEGER);
                lineorderType.add("lo_discount", SqlTypeName.INTEGER);
                lineorderType.add("lo_revenue", SqlTypeName.INTEGER);
                lineorderType.add("lo_supplycost", SqlTypeName.INTEGER);

                lineorderType.add("lo_tax", SqlTypeName.INTEGER);
                lineorderType.add("lo_commitdate", SqlTypeName.INTEGER);
//        lineorderType.add("lo_shopmode", SqlTypeName.VARCHAR);
                lineorderType.add("lo_shopmode", SqlTypeName.INTEGER);

                SycldbTable lineorderTable = new SycldbTable("lineorder", lineorderType.build(), 119994746);
                tables.put("lineorder", lineorderTable);

                RelDataTypeFactory.Builder partType = new RelDataTypeFactory.Builder(typeFactory);
                partType.add("p_partkey", SqlTypeName.INTEGER);
//        partType.add("p_name", SqlTypeName.VARCHAR);
//        partType.add("p_mfgr", SqlTypeName.VARCHAR);
//        partType.add("p_category", SqlTypeName.VARCHAR);
//        partType.add("p_brand1", SqlTypeName.VARCHAR);
//        partType.add("p_color", SqlTypeName.VARCHAR);
//        partType.add("p_type", SqlTypeName.VARCHAR);
                partType.add("p_name", SqlTypeName.INTEGER);
                partType.add("p_mfgr", SqlTypeName.INTEGER);
                partType.add("p_category", SqlTypeName.INTEGER);
                partType.add("p_brand1", SqlTypeName.INTEGER);
                partType.add("p_color", SqlTypeName.INTEGER);
                partType.add("p_type", SqlTypeName.INTEGER);

                partType.add("p_size", SqlTypeName.INTEGER);
//        partType.add("p_container", SqlTypeName.VARCHAR);
                partType.add("p_container", SqlTypeName.INTEGER);

                SycldbTable partTable = new SycldbTable("part", partType.build(), 1000000);
                tables.put("part", partTable);

                RelDataTypeFactory.Builder supplierType = new RelDataTypeFactory.Builder(typeFactory);
                supplierType.add("s_suppkey", SqlTypeName.INTEGER);
//        supplierType.add("s_name", SqlTypeName.VARCHAR);
//        supplierType.add("s_address", SqlTypeName.VARCHAR);
//        supplierType.add("s_city", SqlTypeName.VARCHAR);
//        supplierType.add("s_nation", SqlTypeName.VARCHAR);
//        supplierType.add("s_region", SqlTypeName.VARCHAR);
//        supplierType.add("s_phone", SqlTypeName.VARCHAR);
                supplierType.add("s_name", SqlTypeName.INTEGER);
                supplierType.add("s_address", SqlTypeName.INTEGER);
                supplierType.add("s_city", SqlTypeName.INTEGER);
                supplierType.add("s_nation", SqlTypeName.INTEGER);
                supplierType.add("s_region", SqlTypeName.INTEGER);
                supplierType.add("s_phone", SqlTypeName.INTEGER);

                SycldbTable supplierTable = new SycldbTable("supplier", supplierType.build(), 40000);
                tables.put("supplier", supplierTable);

                RelDataTypeFactory.Builder customerType = new RelDataTypeFactory.Builder(typeFactory);
                customerType.add("c_custkey", SqlTypeName.INTEGER);
//        customerType.add("c_name", SqlTypeName.VARCHAR);
//        customerType.add("c_address", SqlTypeName.VARCHAR);
//        customerType.add("c_city", SqlTypeName.VARCHAR);
//        customerType.add("c_nation", SqlTypeName.VARCHAR);
//        customerType.add("c_region", SqlTypeName.VARCHAR);
//        customerType.add("c_phone", SqlTypeName.VARCHAR);
//        customerType.add("c_mktsegment", SqlTypeName.VARCHAR);
                customerType.add("c_name", SqlTypeName.INTEGER);
                customerType.add("c_address", SqlTypeName.INTEGER);
                customerType.add("c_city", SqlTypeName.INTEGER);
                customerType.add("c_nation", SqlTypeName.INTEGER);
                customerType.add("c_region", SqlTypeName.INTEGER);
                customerType.add("c_phone", SqlTypeName.INTEGER);
                customerType.add("c_mktsegment", SqlTypeName.INTEGER);

                SycldbTable customerTable = new SycldbTable("customer", customerType.build(), 600000);
                tables.put("customer", customerTable);

                RelDataTypeFactory.Builder ddateType = new RelDataTypeFactory.Builder(typeFactory);
                ddateType.add("d_datekey", SqlTypeName.INTEGER);
//        ddateType.add("d_date", SqlTypeName.VARCHAR);
//        ddateType.add("d_dayofweek", SqlTypeName.VARCHAR);
//        ddateType.add("d_month", SqlTypeName.VARCHAR);
                ddateType.add("d_date", SqlTypeName.INTEGER);
                ddateType.add("d_dayofweek", SqlTypeName.INTEGER);
                ddateType.add("d_month", SqlTypeName.INTEGER);

                ddateType.add("d_year", SqlTypeName.INTEGER);
                ddateType.add("d_yearmonthnum", SqlTypeName.INTEGER);
//        ddateType.add("d_yearmonth", SqlTypeName.VARCHAR);
                ddateType.add("d_yearmonth", SqlTypeName.INTEGER);

                ddateType.add("d_daynuminweek", SqlTypeName.INTEGER);
                ddateType.add("d_daynuminmonth", SqlTypeName.INTEGER);
                ddateType.add("d_daynuminyear", SqlTypeName.INTEGER);
                ddateType.add("d_monthnuminyear", SqlTypeName.INTEGER);
                ddateType.add("d_weeknuminyear", SqlTypeName.INTEGER);
//        ddateType.add("d_sellingseasin", SqlTypeName.VARCHAR);
                ddateType.add("d_sellingseasin", SqlTypeName.INTEGER);

                ddateType.add("d_lastdayinweekfl", SqlTypeName.INTEGER);
                ddateType.add("d_lastdayinmonthfl", SqlTypeName.INTEGER);
                ddateType.add("d_holidayfl", SqlTypeName.INTEGER);
                ddateType.add("d_weekdayfl", SqlTypeName.INTEGER);

                SycldbTable ddateTable = new SycldbTable("ddate", ddateType.build(), 2556);
                tables.put("ddate", ddateTable);

                break;
            }
        }

        rootSchema = CalciteSchema.createRootSchema(true);
        tables.forEach(rootSchema::add);
    }

    public RelDataTypeFactory getTypeFactory() {
        return typeFactory;
    }

    public CalciteSchema getRootSchema() {
        return rootSchema;
    }

    @Override
    protected Map<String, Table> getTableMap() {
        return tables;
    }
}
