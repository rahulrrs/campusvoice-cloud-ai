import { useQuery } from "@tanstack/react-query";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { complaintsApi } from "@/integrations/aws/client";

const FAQ = () => {
  const { data: faqs = [], isLoading } = useQuery({
    queryKey: ["faq"],
    queryFn: () => complaintsApi.getFaq(),
  });

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1">
        <section className="border-b bg-card">
          <div className="container mx-auto px-4 py-10">
            <h1 className="text-3xl font-bold text-foreground">Frequently Asked Questions</h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
              Quick answers about submitting complaints, staying anonymous, and tracking updates.
            </p>
          </div>
        </section>

        <section className="container mx-auto px-4 py-10">
          <div className="mx-auto max-w-3xl rounded-2xl border bg-card p-6 shadow-card">
            {isLoading ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
              </div>
            ) : (
              <Accordion type="single" collapsible className="w-full">
                {faqs.map((item) => (
                  <AccordionItem key={item.id} value={item.id}>
                    <AccordionTrigger className="text-left text-base font-semibold">
                      {item.question}
                    </AccordionTrigger>
                    <AccordionContent className="text-sm leading-7 text-muted-foreground">
                      {item.answer}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            )}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default FAQ;
