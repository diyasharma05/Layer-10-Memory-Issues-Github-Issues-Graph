def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Evidence) REQUIRE v.evidence_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artifact) REQUIRE a.artifact_id IS UNIQUE")
