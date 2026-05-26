package com.eurecom.calcite;

import org.apache.calcite.linq4j.AbstractEnumerable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.Linq4j;

import java.util.List;

// TODO: implement with actual data from sycldb
public class SycldbEnumerable extends AbstractEnumerable<Object[]> {
    private final List<Object[]> data;

    public SycldbEnumerable(List<Object[]> data) {
        this.data = data;
    }

    @Override
    public Enumerator<Object[]> enumerator() {
        return Linq4j.enumerator(data);
    }
}
