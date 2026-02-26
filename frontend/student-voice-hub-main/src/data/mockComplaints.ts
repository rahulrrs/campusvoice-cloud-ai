export interface Complaint {
  id: string;
  title: string;
  description: string;
  category: string;
  status: "pending" | "in-progress" | "resolved" | "rejected";
  date: string;
  priority: "low" | "medium" | "high";
}

export const mockComplaints: Complaint[] = [
  {
    id: "CMP001",
    title: "Library AC not working properly",
    description: "The air conditioning in the main library reading hall has been malfunctioning for the past week, making it difficult to study during afternoon hours.",
    category: "Facilities",
    status: "in-progress",
    date: "Feb 3, 2025",
    priority: "medium",
  },
  {
    id: "CMP002",
    title: "WiFi connectivity issues in Hostel Block B",
    description: "The WiFi connection in Block B drops frequently, especially during evening hours when most students need to attend online classes or submit assignments.",
    category: "Technical",
    status: "pending",
    date: "Feb 2, 2025",
    priority: "high",
  },
  {
    id: "CMP003",
    title: "Course registration deadline extension request",
    description: "Due to the technical issues with the portal, many students were unable to complete their course registration before the deadline. Request for a 2-day extension.",
    category: "Academic",
    status: "resolved",
    date: "Jan 28, 2025",
    priority: "high",
  },
  {
    id: "CMP004",
    title: "Cafeteria food quality concerns",
    description: "The quality of food in the main cafeteria has degraded significantly. Multiple students have reported stale food being served.",
    category: "Facilities",
    status: "pending",
    date: "Feb 1, 2025",
    priority: "medium",
  },
  {
    id: "CMP005",
    title: "Missing grade for CS101 final exam",
    description: "My grade for the CS101 final examination conducted on January 15th has not been updated in the portal despite it being over two weeks.",
    category: "Academic",
    status: "resolved",
    date: "Jan 25, 2025",
    priority: "high",
  },
  {
    id: "CMP006",
    title: "Parking space allocation issue",
    description: "Despite having a valid parking permit, I'm consistently unable to find parking space near the engineering block during morning hours.",
    category: "Administrative",
    status: "rejected",
    date: "Jan 20, 2025",
    priority: "low",
  },
];

export const categories = [
  "Academic",
  "Facilities",
  "Technical",
  "Administrative",
  "Hostel",
  "Library",
  "Sports",
  "Other",
];
