import { openDB } from "idb";

export type PendingComplaint = {
  localId: string;
  createdAt: number;
  status: "PENDING" | "SYNCED" | "FAILED";
  data: any;
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

export async function savePendingComplaint(data: any) {
  const db = await dbPromise;

  // ✅ safe ID generator (works everywhere)
  const localId =
    (globalThis.crypto as any)?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

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