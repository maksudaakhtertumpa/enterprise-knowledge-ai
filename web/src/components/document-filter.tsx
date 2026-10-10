"use client";

import { Filter } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { DocumentOut } from "@/lib/types";

export function DocumentFilter({
  documents,
  selected,
  onChange,
}: {
  documents: DocumentOut[];
  selected: string[];
  onChange: (ids: string[]) => void;
}) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" disabled={documents.length === 0}>
          <Filter />
          {selected.length ? `${selected.length} document(s)` : "All documents"}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="max-h-80 w-80 overflow-y-auto">
        <p className="mb-2 text-sm font-medium">Only use these documents</p>
        <div className="space-y-2">
          {documents.map((d) => (
            <div key={d.id} className="flex items-center gap-2">
              <Checkbox
                id={`doc-${d.id}`}
                checked={selected.includes(d.id)}
                onCheckedChange={(on) =>
                  onChange(on ? [...selected, d.id] : selected.filter((x) => x !== d.id))
                }
              />
              <Label htmlFor={`doc-${d.id}`} className="truncate font-normal">
                {d.filename}
              </Label>
            </div>
          ))}
        </div>
        {selected.length > 0 && (
          <Button variant="link" size="sm" className="mt-2 px-0" onClick={() => onChange([])}>
            Clear selection
          </Button>
        )}
      </PopoverContent>
    </Popover>
  );
}
