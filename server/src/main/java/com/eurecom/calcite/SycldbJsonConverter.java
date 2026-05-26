package com.eurecom.calcite;

import com.eurecom.calcite.thrift.*;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.util.List;

public class SycldbJsonConverter {
    private List<RelNode> rels;

    public SycldbJsonConverter(String json) {
        JSONObject obj;
        try {
            obj = (JSONObject) new JSONParser().parse(json);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }

        this.rels = ((List<JSONObject>) obj.get("rels"))
                .stream()
                .map(this::parseRel)
                .toList();
    }

    public List<RelNode> getRels() {
        return rels;
    }

    private RelNode parseRel(JSONObject relObj) {

        long id = Long.parseLong((String) relObj.get("id"));
        String relOp = (String) relObj.get("relOp");
        RelNodeType relOpType = switch (relOp) {
            case "com.eurecom.calcite.SycldbTableScan" -> RelNodeType.TABLE_SCAN;
            case "com.eurecom.calcite.SycldbFilter" -> RelNodeType.FILTER;
            case "com.eurecom.calcite.SycldbProject" -> RelNodeType.PROJECT;
            case "com.eurecom.calcite.SycldbAggregate" -> RelNodeType.AGGREGATE;
            case "com.eurecom.calcite.SycldbJoin" -> RelNodeType.JOIN;
            case "com.eurecom.calcite.SycldbSort" -> RelNodeType.SORT;
            default -> throw new RuntimeException("Unknown RelNodeType: " + relOp);
        };

        RelNode relNode = new RelNode(id, relOpType);

        if (relOpType == RelNodeType.TABLE_SCAN) {
            List<String> tables = (List<String>) relObj.get("table");
            relNode.setTables(tables);
        }

        if (relOpType == RelNodeType.TABLE_SCAN || relOpType == RelNodeType.JOIN) {
            List<Long> inputs = ((List<String>) relObj.get("inputs"))
                    .stream()
                    .map(Long::valueOf)
                    .toList();
            relNode.setInputs(inputs);
        }

        if (relOpType == RelNodeType.FILTER || relOpType == RelNodeType.JOIN) {
            relNode.setCondition(parseExpr((JSONObject) relObj.get("condition")));
        }

        if (relOpType == RelNodeType.JOIN) {
            relNode.setJoinType((String) relObj.get("joinType"));
        }

        if (relOpType == RelNodeType.PROJECT) {
            relNode.setFields(((List<String>) relObj.get("fields")));
            relNode.setExprs(((List<JSONObject>) relObj.get("exprs"))
                    .stream()
                    .map(this::parseExpr)
                    .toList());
        }

        if (relOpType == RelNodeType.AGGREGATE) {
            relNode.setGroup((List<Long>) relObj.get("group"));

            relNode.setAggs(((List<JSONObject>) relObj.get("aggs"))
                    .stream()
                    .map(this::parseAgg)
                    .toList());
        }

        if (relOpType == RelNodeType.SORT) {
            relNode.setCollation(((List<JSONObject>) relObj.get("collation"))
                    .stream()
                    .map(this::parseCollation)
                    .toList());
        }

        return relNode;
    }

    private CollationType parseCollation(JSONObject obj) {
        return new CollationType(
                (Long) obj.get("field"),
                switch ((String) obj.get("direction")) {
                    case "ASCENDING" -> DirectionOption.ASCENDING;
                    case "DESCENDING" -> DirectionOption.DESCENDING;
                    default -> throw new RuntimeException("Unknown CollationType: " + obj.get("direction"));
                },
                switch ((String) obj.get("nulls")) {
                    case "FIRST" -> NullsOption.FIRST;
                    case "LAST" -> NullsOption.LAST;
                    default -> throw new RuntimeException("Unknown CollationType: " + obj.get("nulls"));
                }
        );
    }

    private AggType parseAgg(JSONObject obj) {
        return new AggType(
                (String) ((JSONObject) obj.get("agg")).get("name"),
                (List<Long>) obj.get("operands"),
                (String) obj.get("name"),
                (String) ((JSONObject) obj.get("type")).get("type"),
                (boolean) obj.get("distinct")
        );
    }

    private ExprType parseExpr(JSONObject obj) {
        ExprType exprType;

        // column
        if (obj.containsKey("input")) {
            exprType = new ExprType(ExprOption.COLUMN);

            exprType.setName((String) obj.get("name"));
            exprType.setInput((Long) obj.get("input"));
        }
        // expr
        else if (obj.containsKey("op")) {
            exprType = new ExprType(ExprOption.EXPR);

            exprType.setOp((String) ((JSONObject) obj.get("op")).get("name"));
            exprType.setOperands(((List<JSONObject>) obj.get("operands"))
                    .stream()
                    .map(this::parseExpr)
                    .toList());
        }
        // literal
        else {
            exprType = new ExprType(ExprOption.LITERAL);

            Object literal = obj.get("literal");
            if (literal instanceof Long literalValue) {
                LiteralType literalType = new LiteralType(LiteralOption.LITERAL);
                literalType.setValue(literalValue);
                exprType.setLiteral(literalType);
            } else {
                List<List<String>> rangeSet = (List<List<String>>) ((JSONObject) literal).get("rangeSet");
                LiteralType literalType = new LiteralType(LiteralOption.RANGE);
                literalType.setRangeSet(rangeSet);
                exprType.setLiteral(literalType);
            }
        }

        if (obj.containsKey("type")) {
            exprType.setType((String) ((JSONObject) obj.get("type")).get("type"));
        }

        return exprType;
    }
}
