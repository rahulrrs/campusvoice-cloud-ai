import { openDB } from "idb";

export type QueuedAttachment = {
  name: string;
  type: string;
  size: number;
  file: Blob;
};

export type PendingComplaintData = {
  title: string;
  description: string;
  user_id: string;
  category?: string;
  priority?: string;
  attachment_keys?: string[];
  queued_attachments?: QueuedAttachment[];
};

export type PendingComplaint = {
  localId: string;
  createdAt: number;
  status: "PENDING" | "SYNCED" | "FAILED";
  data: PendingComplaintData;
};

export const dbPromise = openDB("campusvoice", 1, {
  upgrade(db) {
    if (!db.objectStoreNames.contains("pendingComplaints")) {
      db.createObjectStore("pendingComplaints", { keyPath: "localId" });
    }
    if (!db.objectStoreNames.contains("cachedComplaints")) {
      db.createObjectStore("cachedComplaints", { keyPath: "id" });
    }
  },
});

export async function savePendingComplaint(data: PendingComplaintData) {
  const db = await dbPromise;
  const localId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

  await db.put("pendingComplaints", {
    localId,
    createdAt: Date.now(),
    status: "PENDING",
    data,
  } satisfies PendingComplaint);

  return localId;
}

export async function getPendingComplaints(): Promise<PendingComplaint[]> {
  const db = await dbPromise;
  return db.getAll("pendingComplaints");
}

export async function deletePendingComplaint(localId: string) {
  const db = await dbPromise;
  await db.delete("pendingComplaints", localId);
}
