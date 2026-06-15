-- demo_sanitize.sql — SANITIZE the demo database (date-cut <2025-01-01 + anonymize names).
-- RUN ONLY against aps_printing_demo — NEVER prod. Self-refuses on the prod DB name (guard below).
-- Drift-tolerant: every table/column is existence-guarded, so a schema drift skips that target
-- (it does NOT silently leave a real name — the verify gate G1 greps the dump vs name_map.csv).
-- One transaction, ON_ERROR_STOP. Usage (PowerShell):
--   $env:PGPASSWORD="<pw>"
--   & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h <host> -U <user> -d aps_printing_demo `
--       -1 -v ON_ERROR_STOP=1 -f fly_demo\demo_sanitize.sql
-- Materializes RUNBOOK.md Part D (D2 date-cut + D3 anonymization) as a single runnable file.

\echo '== demo_sanitize: START =='
BEGIN;
SET search_path TO aps;

-- ---------- HARD SAFETY: never run on prod ----------
DO $g$
BEGIN
  IF current_database() = 'aps_printing' THEN
    RAISE EXCEPTION 'REFUSING to sanitize the PRODUCTION database "aps_printing". Run only against aps_printing_demo.';
  END IF;
END $g$;

-- ---------- target config (drift-tolerant; everything existence-guarded) ----------
CREATE TEMP TABLE _order_child(tbl text, col text) ON COMMIT DROP;
INSERT INTO _order_child(tbl,col) VALUES
  ('order_materials','01_order_id'),
  ('order_materials_extra','01_order_id'),
  ('material_predictions','01_order_id'),
  ('material_predictions_approved','01_order_id'),
  ('predictions','order_id'),
  ('roll_allocations','02_order_id'),
  ('deficit_confirmations','02_order_id');

CREATE TEMP TABLE _date_cut(tbl text, col text) ON COMMIT DROP;   -- orders handled LAST, separately
INSERT INTO _date_cut(tbl,col) VALUES
  ('documents','00_record_date'),
  ('shipments','1_data_vidvantazhennia'),
  ('shipments_samples','1_data_vidvantazhennia');

CREATE TEMP TABLE _truncate(tbl text) ON COMMIT DROP;  -- derived projections + change-tracking (FK-safe via CASCADE)
INSERT INTO _truncate(tbl) VALUES ('proposed_actions'),('allocation_intents'),('etl_runs');

CREATE TEMP TABLE _name_targets(tbl text, col text, pool text) ON COMMIT DROP;
INSERT INTO _name_targets(tbl,col,pool) VALUES
  ('orders','01_customer_name','client'),
  ('orders','03_corporation_name','client'),
  ('orders','05_manufacturer_name','client'),
  ('orders','58_manager_name','staff'),
  ('documents','13_customer_name','client'),
  ('documents','21_account_name','client'),
  ('inventory','01_customer_name','client'),
  ('inventory','03_corporation_name','client'),
  ('inventory','05_payer_name','client'),
  ('inventory','52_manager_name','staff'),
  ('shipments','5_platnyk_naymenuvannia','client'),
  ('shipments','14_zamovnyk_nazva_1','client'),
  ('shipments','15_zamovnyk_nazva_2','client'),
  ('shipments','22_korporatsiia_nazva','client'),
  ('shipments','29_fop','client'),
  ('shipments','83_menedzher','staff'),
  ('shipments_samples','14_zamovnyk_nazva_1','client'),
  ('shipments_samples','s_otrymuvach_nazva','client'),
  ('inventory_balances','13_postachalnyk_nazva','supplier'),
  ('inventory_balances','17_zamovnyk_nazva','client');

CREATE TEMP TABLE _freetext(tbl text, col text) ON COMMIT DROP;  -- may embed a name/brand/city -> NULL
INSERT INTO _freetext(tbl,col) VALUES
  ('shipments','34_produkt_1'),('shipments','35_produkt_2'),('shipments','41_dyzain'),
  ('shipments','43_brend_hrupa_produktsii_zamovnyka'),('shipments','28_misto'),
  ('orders','12_order_name'),('orders','16_design_studio'),
  ('documents','15_order_group_name'),('documents','17_order_name'),('documents','23_nomenclature_name'),
  ('inventory','07_contract_name'),('inventory','09_spec_name'),('inventory','22_design_name'),
  ('inventory_balances','37_zamovlennia_nazva'),('inventory_balances','56_naimenuvannia_zamovlennia'),
  ('inventory_balances','2_nomenklatura_nazva'),
  ('predictions','order_name');

CREATE OR REPLACE FUNCTION pg_temp._has(tbl text, col text) RETURNS boolean AS $f$
  SELECT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema='aps' AND table_name=tbl AND column_name=col);
$f$ LANGUAGE sql;

-- ========================= D2: DATE CUT (keep rows < 2025-01-01; NULL order-date kept) =========================
CREATE TEMP TABLE cut_orders(oid bigint) ON COMMIT DROP;
DO $$
BEGIN
  IF pg_temp._has('orders','10_order_id') AND pg_temp._has('orders','11_order_date') THEN
    EXECUTE 'INSERT INTO cut_orders SELECT "10_order_id" FROM aps.orders WHERE "11_order_date" >= DATE ''2025-01-01''';
  END IF;
END $$;

DO $$  -- child/inference rows of cut orders
DECLARE r record;
BEGIN
  FOR r IN SELECT tbl,col FROM _order_child LOOP
    IF pg_temp._has(r.tbl, r.col) THEN
      EXECUTE format('DELETE FROM aps.%I WHERE %I IN (SELECT oid FROM cut_orders)', r.tbl, r.col);
    END IF;
  END LOOP;
END $$;

DO $$  -- top-level fact rows by own date col, orders LAST
DECLARE r record;
BEGIN
  FOR r IN SELECT tbl,col FROM _date_cut LOOP
    IF pg_temp._has(r.tbl, r.col) THEN
      EXECUTE format('DELETE FROM aps.%I WHERE %I >= DATE ''2025-01-01''', r.tbl, r.col);
    END IF;
  END LOOP;
  IF pg_temp._has('orders','11_order_date') THEN
    EXECUTE 'DELETE FROM aps.orders WHERE "11_order_date" >= DATE ''2025-01-01''';
  END IF;
END $$;

DO $$  -- TRUNCATE derived projections (FK to etl_runs has no CASCADE -> plain DELETE would FK-violate)
DECLARE r record; present text[] := '{}';
BEGIN
  FOR r IN SELECT tbl FROM _truncate LOOP
    IF to_regclass('aps.'||r.tbl) IS NOT NULL THEN present := present || ('aps.'||quote_ident(r.tbl)); END IF;
  END LOOP;
  IF array_length(present,1) IS NOT NULL THEN
    EXECUTE 'TRUNCATE '||array_to_string(present,', ')||' CASCADE';
  END IF;
END $$;

-- ========================= D3: ANONYMIZE NAMES (base tables only; views recompute) =========================
CREATE TEMP TABLE _names(real_name text, pool text) ON COMMIT DROP;
DO $$
DECLARE r record;
BEGIN
  FOR r IN SELECT tbl,col,pool FROM _name_targets LOOP
    IF pg_temp._has(r.tbl, r.col) THEN
      EXECUTE format('INSERT INTO _names SELECT DISTINCT %I, %L FROM aps.%I WHERE %I IS NOT NULL AND btrim(%I::text) <> ''''',
                     r.col, r.pool, r.tbl, r.col, r.col);
    END IF;
  END LOOP;
END $$;

CREATE TEMP TABLE name_map ON COMMIT DROP AS
SELECT real_name, pool,
  (CASE pool WHEN 'client' THEN 'Демо-клієнт ' WHEN 'supplier' THEN 'Демо-постачальник ' ELSE 'Демо-менеджер ' END
   || row_number() OVER (PARTITION BY pool ORDER BY real_name))::text AS fake_name
FROM (SELECT DISTINCT real_name, pool FROM _names) s;

-- export real->fake map OUTSIDE the DB for the verify gate (G1 greps the dump vs the real_name column).
-- The agent MUST preserve this file until AFTER verify, then delete at final teardown.
\copy (SELECT real_name, pool, fake_name FROM name_map ORDER BY pool, fake_name) TO 'C:/demo_build/name_map.csv' CSV HEADER

DO $$  -- apply replacement to every existing name column (same real -> same fake)
DECLARE r record;
BEGIN
  FOR r IN SELECT tbl,col,pool FROM _name_targets LOOP
    IF pg_temp._has(r.tbl, r.col) THEN
      EXECUTE format('UPDATE aps.%I t SET %I = m.fake_name FROM name_map m WHERE t.%I = m.real_name AND m.pool = %L',
                     r.tbl, r.col, r.col, r.pool);
    END IF;
  END LOOP;
END $$;

DO $$  -- null free-text fields
DECLARE r record;
BEGIN
  FOR r IN SELECT tbl,col FROM _freetext LOOP
    IF pg_temp._has(r.tbl, r.col) THEN
      EXECUTE format('UPDATE aps.%I SET %I = NULL WHERE %I IS NOT NULL', r.tbl, r.col, r.col);
    END IF;
  END LOOP;
END $$;

-- ========================= SANITY (pre-COMMIT; eyeball or assert in the wrapper) =========================
\echo '== row counts + max dates (must be < 2025-01-01; orders may be NULL) =='
SELECT 'orders' AS t, count(*) n, max("11_order_date") max_date FROM aps.orders
UNION ALL SELECT 'documents', count(*), max("00_record_date") FROM aps.documents
UNION ALL SELECT 'shipments', count(*), max("1_data_vidvantazhennia") FROM aps.shipments;
\echo '== fake-name pools generated =='
SELECT pool, count(*) AS fakes FROM name_map GROUP BY pool ORDER BY pool;

COMMIT;
\echo '== demo_sanitize: COMMIT OK =='
