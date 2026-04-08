import type { ComplaintRecord } from "@/integrations/aws/client";
import QueueRow from "@/components/admin/QueueRow";

type AdminQueueProps = {
  complaints: ComplaintRecord[];
  selectedComplaintId: string | null;
  onSelect: (complaintId: string) => void;
};

const AdminQueue = ({ complaints, selectedComplaintId, onSelect }: AdminQueueProps) => {
  if (!complaints.length) {
    return (
      <div className="rounded-[20px] border border-dashed border-slate-300 bg-white/90 p-8 text-center shadow-sm">
        <p className="text-base font-semibold text-slate-950">No complaints in this view</p>
        <p className="mt-2 text-sm text-slate-600">Clear a filter or adjust the search to reopen the queue.</p>
      </div>
    );
  }

  return (
    <div className="max-h-[calc(100vh-13rem)] space-y-2 overflow-y-auto pr-1">
      {complaints.map((complaint) => (
        <QueueRow
          key={complaint.id}
          complaint={complaint}
          selected={complaint.id === selectedComplaintId}
          onSelect={() => onSelect(complaint.id)}
        />
      ))}
    </div>
  );
};

export default AdminQueue;
