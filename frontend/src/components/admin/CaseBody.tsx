import type { ComplaintRecord } from "@/integrations/aws/client";

type CaseBodyProps = {
  complaint: ComplaintRecord;
};

const CaseBody = ({ complaint }: CaseBodyProps) => {
  return (
    <section className="rounded-[22px] border border-slate-200/80 bg-white px-5 py-4">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Complaint</p>
      <p className="mt-2 max-w-4xl text-sm leading-7 text-slate-700">{complaint.description}</p>
    </section>
  );
};

export default CaseBody;
