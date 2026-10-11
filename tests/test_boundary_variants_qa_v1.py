"""Meaningful offline C admission/SDK selection/plain QA tests; no live models."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import threading
from types import CodeType
import unittest
from unittest.mock import patch

from tools import run_boundary_variants_qa_v1 as d
from tools import boundary_variants_qa_sources_v1 as s
from tools import audit_boundary_variants_banks_v1 as a
from tools import run_regroup_refinement_qa_v1 as old
from tests.test_sdk20_plain_qa_v1 import job as old_job
from tests.test_sdk20_judge_retest_v1 import Transport, KEYS
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.evaluation_plain_text_v1 import reader_input


ARM="boundary_multiscale"
RAW="c1_multiscale"


def job(qid="c0_q0",ci=0):return old_job(qid,ARM,ci)


class BoundaryQATests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="boundary_qa_offline_");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.run=self.root/"qa";self.run.mkdir()
        for target in ("requests.Session.post","requests.Session.request","socket.create_connection","jev_mem.providers.load_credentials"):
            p=patch(target,side_effect=AssertionError("No real credentials/network"));p.start();self.addCleanup(p.stop)

    def m(self,jobs,phase="firstpass"):
        return {"schema":d.SCHEMA,"protocol_sha256":"mock-"+phase,"arm":ARM,"phase_kind":phase,
                "phase_id":"mock-"+phase,"jobs":jobs,"source_manifest":{"path":"mock","sha256":"mock"}}

    def factory(self,path,ci,m):
        return PlainTextEvaluationAPI(path,keys=KEYS,cache_roots=(),retries=2,
            context={"architecture":d.plain.SCHEMA,"conversation":ci,"phase":"qa","phase_id":m["phase_id"]})

    def execute_body(self,j,outputs):
        m=self.m([j]);t=Transport(outputs)
        with patch.object(d,"execution_manifest",return_value=m),patch("requests.Session",return_value=t):
            out=d._qa_core(self.run,api_factory=self.factory)
        return m,out,t

    def row(self,j,stage):return d.verify(d.plain.stage_files(self.run,j)[stage])

    def test_independent_bindings_and_original_plain_one(self):
        self.assertIs(d.plain.one,old.plain.one)
        self.assertIs(old.sources,d._template.sources)
        self.assertIsNot(d.sources,old.sources)
        self.assertEqual(old.POLICY["slots_per_arm"],400)
        self.assertEqual(d.POLICY["slots_per_arm"],1540)
        for name in d.REBOUND_FUNCTIONS:
            if name not in {"prepare","retrieve","qa","manifest","terminal","main","retrieve_worker","prepare_recovery"}:
                self.assertIs(getattr(d,name).__code__,getattr(old,name).__code__)
        self.assertEqual(d._manifest_core.__code__,old.manifest.__code__)
        self.assertEqual(d._terminal_core.__code__,old.terminal.__code__)

    def test_exact_three_cardinality_ast_edits_only(self):
        for name,field in d.CARDINALITY_FIELDS.items():
            original=getattr(old,name);tree=ast.parse(inspect.getsource(original))
            for node in ast.walk(tree):
                if isinstance(node,ast.Dict):
                    for key,value in zip(node.keys,node.values):
                        if isinstance(key,ast.Constant) and key.value==field:value.value=1540
            namespace={};exec(compile(ast.fix_missing_locations(tree),"test","exec"),original.__globals__,namespace)
            actual=d._cardinality_code(original,field);expected=namespace[name].__code__
            self.assertEqual(actual.co_code,expected.co_code)
            def code_shape(code):
                return (code.co_code,code.co_names,tuple(code_shape(v) if isinstance(v,CodeType) else v for v in code.co_consts))
            self.assertEqual(code_shape(actual),code_shape(expected))
            self.assertEqual(actual.co_names,expected.co_names)
            with self.assertRaisesRegex(ValueError,"Exactly one"):
                d._cardinality_code(original,"undeclared_cardinality")

    def test_real_full_cohort_lock_all_payload_fields(self):
        rows,dates=s.cohort();self.assertEqual(sum(map(len,rows.values())),1540)
        self.assertEqual(set(rows),{str(ci) for ci in range(10)});self.assertEqual(len(dates),10)
        lock=s.verify_lock()
        actual=[dict(conversation=ci,query_id=q["id"],source_qa_index=q["source_qa_index"],
                     question=q["question"],reference=q["reference"],category=q["category"],reference_date=dates[str(ci)])
                for ci in range(10) for q in rows[str(ci)]]
        self.assertEqual(actual,lock["dataset"]["questions"])
        for key in ("question","reference","category"):
            changed=copy.deepcopy(rows);changed["0"][0][key]="drift"
            with patch.object(s,"verify_lock",return_value=lock),patch.object(s.old,"cohort",return_value=(changed,dates)),self.assertRaisesRegex(ValueError,"full question"):
                s.cohort()
        with patch.object(s,"verify_lock",return_value=lock),patch.object(s.old,"cohort",return_value=(rows,{**dates,"0":"2099-01-01"})),self.assertRaises(ValueError):s.cohort()

    def test_exact_cache_only_embedding_signature_and_missing_fail(self):
        signature={"provider":"gemini","model":"gemini-embedding-001","operation":"embed","body":{"x":1}}
        key=d._request_digest(signature);p=self.root/"cache.json"
        p.write_text(json.dumps({"response":{"embedding":{"values":[1,2]}}}),encoding="utf-8")
        ptr=s.pointer(p);api=d.CacheOnlyEmbeddings({key:{"source":ptr,"target":ptr}})
        with patch.object(d.embedding_pack,"validate_raw",return_value=signature):
            self.assertEqual(api.request("gemini",signature["model"],signature["body"],operation="embed",stage="QUERY"),{"embedding":{"values":[1,2]}})
        self.assertEqual(api.events[0]["model_calls"],0)
        with self.assertRaisesRegex(RuntimeError,"no network fallback"):
            api.request("gemini",signature["model"],{"x":2},operation="embed",stage="QUERY")
        with patch.object(d.embedding_pack,"validate_raw",return_value={}),self.assertRaisesRegex(ValueError,"signature"):
            api.request("gemini",signature["model"],signature["body"],operation="embed",stage="QUERY")
        with self.assertRaises(RuntimeError):api.count_tokens("x")
        self.assertFalse(hasattr(api,"session"))

    def test_sdk20_order_and_reader_whitelist(self):
        j=job();q={"id":j["query_id"],"question":"  Where?  ","reference":2022,"category":2}
        bank={"facts":[dict(f,metadata="invisible") for f in j["selected_facts"]]}
        m={"protocol_sha256":"p","reference_dates":{"0":j["reference_date"]},"banks":{"0":{ARM:{"path":"x","sha256":"y"}}}}
        ids=list(reversed(j["selected_ids"]));path=self.run/f"conv_0/selection/{ARM}/c0_q0.json"
        selection={**d.native_qa.identity(0,ARM,q,m),"bank_sha256":"b","status":"complete","selected_ids":ids,"capture":{"base20_ids":ids}}
        d.save_new(path,selection)
        with patch.object(d.native_qa,"get_bank",return_value=(bank,"b")):
            frozen=d.job_from_selection(self.run,0,ARM,q,m)
        self.assertEqual(frozen["selected_ids"],ids);self.assertEqual(frozen["selected_facts"],list(reversed(j["selected_facts"])))
        self.assertEqual(frozen["question"],"  Where?  ");self.assertEqual(frozen["reference"],2022)
        self.assertTrue(all(set(f)=={"id","text","observation_date"} for f in frozen["selected_facts"]))

    def test_failed_selection_and_extra_reader_metadata_refused(self):
        j=job();q={"id":j["query_id"],"question":j["question"],"reference":j["reference"],"category":j["category"]}
        m={"protocol_sha256":"p"};p=self.run/f"conv_0/selection/{ARM}/c0_q0.json"
        d.save_new(p,{**d.native_qa.identity(0,ARM,q,m),"bank_sha256":"b","status":"failed"})
        with patch.object(d.native_qa,"get_bank",return_value=({"facts":[]},"b")),self.assertRaisesRegex(ValueError,"partial"):
            d.job_from_selection(self.run,0,ARM,q,m)
        j["selected_facts"][0]["source_sidecar"]={}
        with self.assertRaises(ValueError):d.check_job(j,ARM)

    def test_reader_f1_persisted_before_same_judge_configuration(self):
        j=job();j["reference"]=2022;m=self.m([j]);t=Transport(["2022","[[yes]]"]);post=t.post
        def checked_post(url,**kw):
            if "3.5-flash-lite" in url:self.assertEqual(self.row(j,"reader")["result"]["metrics"]["official_locomo_f1"],1.)
            return post(url,**kw)
        t.post=checked_post
        with patch.object(d,"execution_manifest",return_value=m),patch("requests.Session",return_value=t):out=d._qa_core(self.run,api_factory=self.factory)
        self.assertEqual(json.loads(t.calls[0]["contents"][0]["parts"][0]["text"]),reader_input(j["question"],j["selected_facts"],reference_date=j["reference_date"]))
        self.assertEqual(t.calls[0]["generationConfig"],{"temperature":0,"maxOutputTokens":2048,"thinkingConfig":{"thinkingBudget":0}})
        self.assertEqual(t.calls[1]["generationConfig"],{"temperature":0,"maxOutputTokens":2048})
        self.assertEqual(out["planned_per_arm"],1540)

    def test_full1540_controller_dispatch_count_no_models(self):
        jobs=[{"job_id":f"{ARM}/c{ci}_q{k}","conversation":ci,"method":ARM,"query_id":f"c{ci}_q{k}"} for ci in range(10) for k in range(154)]
        m=self.m(jobs);seen=[];lock=threading.Lock()
        def one(run,j,manifest,api,official,error):
            with lock:seen.append(j["job_id"])
        with patch.object(d,"execution_manifest",return_value=m),patch.object(d.plain,"one",side_effect=one),patch.object(d.plain,"stage_files",return_value={}),patch.object(d,"publish_json"),patch.object(d,"seal_firstpass") as seal:
            out=d.qa(self.run,api_factory=lambda *args:object())
        self.assertEqual(len(set(seen)),1540);self.assertEqual(out["total"],1540);seal.assert_called_once()

    def test_partial_firstpass_dispatch_refused_before_api(self):
        with patch.object(d,"execution_manifest",return_value=self.m([job()])),self.assertRaisesRegex(ValueError,"partial"):
            d.qa(self.run,api_factory=lambda *args:self.fail("No clients for partial"))

    def test_valid_no_not_retried_unknown_or_second_recovery_refused(self):
        m,out,_=self.execute_body(job(),["Paris","[[no]]"])
        self.assertEqual(d.recovery_jobs(m,out),[])
        missing=copy.deepcopy(out);missing["results"][0]["judge"]=None
        with self.assertRaisesRegex(ValueError,"Unknown"):d.recovery_jobs(m,missing)
        with self.assertRaisesRegex(ValueError,"Only one"):d.recovery_jobs({**m,"phase_kind":"technical_recovery"},out)

    def test_strict_failed_judge_preserves_reader_for_recovery(self):
        m,out,_=self.execute_body(job(),["Paris","Explanation\n[[yes]]"])
        self.assertEqual(self.row(job(),"reader")["status"],"complete")
        self.assertEqual(self.row(job(),"judge")["status"],"failed")
        candidates=d.recovery_jobs(m,out);self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]["attempt_kind"],"judge_only_after_successful_reader")
        self.assertEqual(candidates[0]["reader_source"],out["results"][0]["reader"])

    def test_firstpass_seal_immutable_and_all1540_required(self):
        for name in ("run_manifest.json","run_outcome.json","source_jobs.json"):d.save_new(self.run/name,{"name":name})
        m=self.m([]);m["source_manifest"]=s.pointer(self.run/"source_jobs.json")
        out={"results":[{"query_id":"q"}],"total":1540,"planned_per_arm":1540}
        with patch.object(d,"terminal",return_value=(m,out)):
            first=d.seal_firstpass(self.run);self.assertEqual(first["total"],1540)
            self.assertEqual(d.seal_firstpass(self.run),first)
            with patch.object(d,"terminal",return_value=(m,{**out,"results":[]})),self.assertRaisesRegex(ValueError,"changed"):
                d.seal_firstpass(self.run)
        with patch.object(d,"_terminal_core",return_value=(m,{**out,"total":1539})),self.assertRaisesRegex(ValueError,"all1540"):
            d.terminal(self.run)

    def gate_fixture(self,mutate=None):
        source=self.root/"writer";source.mkdir();manifest=source/"run_manifest.json";d.save_new(manifest,{"fixture":True})
        mp=s.pointer(manifest);inputs=[];inspected=[];gate_rows=[];inv={"frozen-native":"digest"}
        for ci in range(10):
            folder=source/f"conv_{ci}"/RAW;native=folder/"native";native.mkdir(parents=True)
            ip=folder/"group_inputs.json";d.save_new(ip,{"envelopes":[{}]});inp=s.pointer(ip);inputs.append({"conversation":ci,"arm":RAW,"input":inp})
            jp=folder/"journal.json";d.save_new(jp,{"proof":True})
            bank={"complete":True,"source_dataset_sha256":s.SOURCE_HASH,"conversation":ci,"arm":RAW,"protocol_sha256":"writer-protocol",
                  "group_inputs":inp,"journal_files":[s.pointer(jp)],"facts":job()["selected_facts"],"native_dir":str(native)}
            bp=folder/"bank.json";d.save_new(bp,bank);bankptr=s.pointer(bp)
            state={"status":"complete","finished_utc":"closed","conversation":ci,"arm":RAW,"protocol_sha256":"writer-protocol",
                   "bank":bankptr,"completed_groups":1,"expected_groups":1,"facts":20}
            sp=folder/"write_status.json";d.save_new(sp,state);status=s.pointer(sp)
            inspected.append({"conversation":ci,"arm":RAW,"strict_passed":True,"source_generation_persistence_passed":True,
                              "bank":bankptr,"write_status":status,"group_inputs":inp,"native_inventory":inv})
            gate_rows.append({"conversation":ci,"arm":ARM,"source_arm":RAW,"source_kind":"strict_complete","original_write_status":"complete",
                              "source_manifest":mp,"bank":bankptr,"write_status":status,"group_inputs":inp,"native_inventory":inv})
        report={"schema":a.SCHEMA,"strict_passed":True,"conversations":list(range(10)),"arm":RAW,"writer_manifest":mp,"limitations":[],
                "code_sha256":{p:d.sha(d.ROOT/p) for p in a.CODE},"source_file_hashes":{},"audit_tool":s.pointer(a.__file__),
                "writer_scheduling":{"mode":"serial_closed","controllers":[]},"rows":inspected}
        rp=self.root/"audit.json";d.save_new(rp,report);auditptr=s.pointer(rp)
        for row in gate_rows:row["audits"]={"source_date":[auditptr],"index":[auditptr]}
        gate={"schema":s.GATE_SCHEMA,"status":"passed","arm":ARM,"source_arm":RAW,"source_manifest":mp,"audit":auditptr,"limitations":[],
              "checks":{k:True for k in ("ten_complete_banks","source_dates_and_generation_chain_verified","current_sdk20_index_dependencies_valid","content_quality_findings_disclosed")},"banks":gate_rows}
        if mutate:mutate(gate)
        gp=self.root/"gate.json";d.save_new(gp,gate)
        parent={"arms":[RAW],"protocol_sha256":"writer-protocol","inputs":inputs,"boundary_variant":{"arm":RAW,"resources":{"snapshot":"fixture","files_sha256":{}}}}
        return gp,parent,inv

    def admit_fixture(self,gp,parent,inv):
        with patch.object(s.writer,"manifest",return_value=parent),patch.object(s,"inventory",return_value=inv),patch.object(s.old,"validate_resources"),patch.object(a,"closed_controllers",return_value={"mode":"serial_closed","controllers":[]}):
            return s.admit_gate(gp)

    def test_full10_gate_admits_exact_source_arm_to_new_qa_arm(self):
        gp,parent,inv=self.gate_fixture();gate,banks,_=self.admit_fixture(gp,parent,inv)
        self.assertEqual(gate["arm"],ARM);self.assertEqual(set(banks),{(ci,ARM) for ci in range(10)})
        self.assertTrue(all(b["bank"]["arm"]==RAW for b in banks.values()))
        with patch.object(s,"inventory",return_value={"changed":"native"}):
            # Separate explicit mismatch assertion (normal helper mocks inventory).
            with patch.object(s.writer,"manifest",return_value=parent),patch.object(s.old,"validate_resources"),patch.object(a,"closed_controllers",return_value={"mode":"serial_closed","controllers":[]}),self.assertRaisesRegex(ValueError,"inventory"):
                s.admit_gate(gp)

    def test_failed_or_partial_gate_cannot_erase_original_status(self):
        gp,parent,inv=self.gate_fixture(lambda g:g["banks"][0].update(source_kind="reviewed_export",original_write_status="failed"))
        with self.assertRaisesRegex(ValueError,"failure/export"):self.admit_fixture(gp,parent,inv)

    def test_missing_index_audit_and_duplicate_cohort_blocked(self):
        gp,parent,inv=self.gate_fixture(lambda g:g["banks"][0]["audits"].pop("index"))
        with self.assertRaisesRegex(ValueError,"Both complete"):self.admit_fixture(gp,parent,inv)
        changed=d.verify(gp);changed["banks"][1]=copy.deepcopy(changed["banks"][0]);changed.pop("content_sha256")
        p=self.root/"duplicate.json";d.save_new(p,changed)
        with self.assertRaisesRegex(ValueError,"distinct"):self.admit_fixture(p,parent,inv)

    def test_dev_report_never_generates_qa_gate(self):
        p=self.root/"dev.json";d.save_new(p,{"schema":a.SCHEMA,"strict_passed":True,"conversations":[0,1]})
        with self.assertRaisesRegex(ValueError,"full ten"):a.write_gate(p,self.root/"gate.json")

    def test_prepare_holds_existing_source_locks_through_copy(self):
        gp,parent,inv=self.gate_fixture();writer_run=Path(d.verify(gp)["source_manifest"]["path"]).parent
        # Establish real old lock files, then release them before prepare.
        with a.boundary.global_runner.exclusive_driver(writer_run.parents[1]),a.writer.writer_process_lock(writer_run):pass
        def copy_probe(run,gate):
            with self.assertRaises(RuntimeError):
                with a.writer.writer_process_lock(writer_run):self.fail("Copy must exclude original Writer")
            return {"model_calls":0}
        with patch.object(d,"_prepare_core",side_effect=copy_probe):
            self.assertEqual(d.prepare(self.run,gp),{"model_calls":0})
        with a.writer.writer_process_lock(writer_run):pass

    def test_live_writer_lock_and_active_controller_blocked(self):
        run=self.root/"parent"/RAW/"writer";run.mkdir(parents=True)
        parent=run.parents[1]
        with a.boundary.global_runner.exclusive_driver(parent),a.writer.writer_process_lock(run):
            with self.assertRaisesRegex(ValueError,"lock"):a.closed_controllers(run,RAW,[0,1])
        # Both old lock contexts leave files behind. Released bytes are legal.
        self.assertTrue((run/"writer_execution.lock").exists());self.assertTrue((parent/"driver.lock").exists())
        before={p:p.read_bytes() for p in (run/"writer_execution.lock",parent/"driver.lock")}
        self.assertEqual(a.closed_controllers(run,RAW,[0,1]),{"mode":"serial_closed","controllers":[]})
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        with a.source_locks(run),a.source_locks(run):
            self.assertEqual(a.closed_controllers(run,RAW,[0,1])["mode"],"serial_closed")
        controller=run.parents[1]/"parallel_controller_v1/write_development";controller.mkdir(parents=True)
        d.save_new(controller/"manifest.json",{"schema":a.parallel.VERSION,"parent":{"mock":True},
            "policy":{"stage":"write","arms":[RAW],"conversations":[0,1]},"code_sha256":{p:d.sha(d.ROOT/p) for p in a.parallel.FILES}})
        d.save_new(controller/"parent_state.json",{"status":"running"})
        with patch.object(a.boundary,"pointer",return_value={"mock":True}),self.assertRaisesRegex(ValueError,"Active"):
            a.closed_controllers(run,RAW,[0,1])


if __name__=="__main__":unittest.main()
