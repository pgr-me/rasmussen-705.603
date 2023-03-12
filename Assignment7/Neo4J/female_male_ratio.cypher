LOAD CSV WITH HEADERS FROM "file:///SpeedDatingData.csv" AS row
    WITH row 
    WHERE
        NOT row.age IS null AND
        NOT row.iid IS null AND
        NOT row.pid IS null AND
        NOT row.age_o IS null and
        NOT row.race IS null AND
        NOT row.race_o IS null AND
        NOT row.match IS null AND
        NOT row.int_corr IS null AND
        NOT row.gender IS null AND
        NOT row.samerace IS null AND
        NOT row.age_o IS null
    MERGE(
        p1 :Person {
            id:row.iid,
            age:toInteger(row.age),
            race:toInteger(row.race)
        }
    )
    MERGE(
        p2: Person {
            id:row.pid,
            age:toInteger(row.age_o),
            race:toInteger(row.race_o)
        }
    )
    MERGE(
        (p1) - [:Date {
            match: toInteger(row.match),
            int_corr: row.int_corr,
            race_diff:toInteger(row.samerace),
            age_diff:abs(toInteger(row.age)- toInteger(row.age_o))
        }] -> (p2)) SET p1.gender = toInteger(row.gender)
;

CALL {
    MATCH (n:Person)
    WHERE n.gender=0
    WITH count(n) AS count
    RETURN count as n_females
}

CALL {
    MATCH (n:Person)
    WHERE n.gender=1
    WITH count(n) AS count
    RETURN count AS n_males
}

RETURN (n_females * 1.0) / (n_males * 1.0);
