import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import api from "@/api/api";
import Loader from "@/components/Loader";
import { CheckCircle2, RefreshCw, Upload } from "lucide-react";

const RetrainModel = () => {
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      toast.error("Please select a CSV file for retraining");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Directly POST the file to /retrain_model (no server path handling required)
      const retrainResp = await api.post("/retrain_model", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(retrainResp.data);
      toast.success("Model retrained successfully");
      setFile(null);
    } catch (error: any) {
      toast.error(error.response?.data?.error || "Failed to retrain model");
      console.error("Retrain error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="dashboard-section">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 rounded-lg water-gradient flex items-center justify-center">
            <RefreshCw className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-foreground">Retrain Model</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Upload new training data to improve leak detection
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="retrain-file">Training Data (CSV)</Label>
            <div className="mt-2">
              <Input
                id="retrain-file"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="cursor-pointer"
              />
            </div>
            {file && (
              <p className="text-sm text-muted-foreground mt-2">
                Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
              </p>
            )}
          </div>

          <div className="bg-muted p-4 rounded-lg">
            <p className="text-sm text-muted-foreground">
              <strong>Note:</strong> Retraining will update the ML model with new data.
              This process may take several minutes depending on dataset size.
            </p>
          </div>

          <Button
            type="submit"
            disabled={loading || !file}
            className="w-full water-gradient gap-2"
          >
            <Upload className="w-4 h-4" />
            {loading ? "Retraining..." : "Start Retraining"}
          </Button>
        </form>
      </Card>

      {loading && <Loader message="Retraining model... This may take a while." />}

      {result && (
        <Card className="p-6 border-success">
          <div className="flex items-start gap-4">
            <CheckCircle2 className="w-8 h-8 text-success flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-xl font-bold mb-2">Retraining Complete</h3>
              <div className="space-y-2 text-sm">
                {result.features_used && (
                  <p>
                    <span className="font-semibold">Features used:</span>{" "}
                    {Array.isArray(result.features_used) ? result.features_used.join(", ") : String(result.features_used)}
                  </p>
                )}
                {result.categorical && result.categorical.length > 0 && (
                  <p>
                    <span className="font-semibold">Categorical columns:</span>{" "}
                    {result.categorical.join(", ")}
                  </p>
                )}
                {result.message && (
                  <p className="text-muted-foreground mt-2">{result.message}</p>
                )}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default RetrainModel;
