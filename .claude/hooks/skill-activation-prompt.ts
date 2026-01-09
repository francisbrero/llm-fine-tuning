#!/usr/bin/env npx tsx

/**
 * Skill Activation Hook for Claude Code
 *
 * Analyzes user prompts and suggests relevant skills based on keyword matching.
 * Runs on UserPromptSubmit hook.
 */

import * as fs from 'fs';
import * as path from 'path';

interface SkillRule {
  type: 'domain' | 'runbook' | 'reference';
  priority: 'critical' | 'high' | 'medium' | 'low';
  file: string;
  description: string;
  promptTriggers: {
    keywords: string[];
    intentPatterns: string[];
  };
}

interface SkillRules {
  version: string;
  skills: Record<string, SkillRule>;
}

interface PromptInput {
  prompt: string;
}

interface MatchedSkill {
  name: string;
  skill: SkillRule;
  matchedKeywords: string[];
  matchedPatterns: string[];
  score: number;
}

function loadSkillRules(): SkillRules {
  const rulesPath = path.join(__dirname, 'skill-rules.json');
  const content = fs.readFileSync(rulesPath, 'utf-8');
  return JSON.parse(content);
}

function matchSkills(prompt: string, rules: SkillRules): MatchedSkill[] {
  const promptLower = prompt.toLowerCase();
  const matches: MatchedSkill[] = [];

  for (const [name, skill] of Object.entries(rules.skills)) {
    const matchedKeywords: string[] = [];
    const matchedPatterns: string[] = [];

    // Check keywords
    for (const keyword of skill.promptTriggers.keywords) {
      if (promptLower.includes(keyword.toLowerCase())) {
        matchedKeywords.push(keyword);
      }
    }

    // Check intent patterns (regex)
    for (const pattern of skill.promptTriggers.intentPatterns) {
      try {
        const regex = new RegExp(pattern, 'i');
        if (regex.test(promptLower)) {
          matchedPatterns.push(pattern);
        }
      } catch {
        // Invalid regex, skip
      }
    }

    // Calculate score
    const keywordScore = matchedKeywords.length * 2;
    const patternScore = matchedPatterns.length * 3;
    const priorityMultiplier = {
      critical: 4,
      high: 3,
      medium: 2,
      low: 1
    }[skill.priority];

    const score = (keywordScore + patternScore) * priorityMultiplier;

    if (matchedKeywords.length > 0 || matchedPatterns.length > 0) {
      matches.push({
        name,
        skill,
        matchedKeywords,
        matchedPatterns,
        score
      });
    }
  }

  // Sort by score descending
  return matches.sort((a, b) => b.score - a.score);
}

function groupByPriority(matches: MatchedSkill[]): Record<string, MatchedSkill[]> {
  const groups: Record<string, MatchedSkill[]> = {
    critical: [],
    high: [],
    medium: [],
    low: []
  };

  for (const match of matches) {
    groups[match.skill.priority].push(match);
  }

  return groups;
}

function formatOutput(matches: MatchedSkill[]): string {
  if (matches.length === 0) {
    return '';
  }

  const grouped = groupByPriority(matches);
  const lines: string[] = [];

  lines.push('');
  lines.push('=== SKILL ACTIVATION ===');
  lines.push('');

  for (const priority of ['critical', 'high', 'medium', 'low']) {
    const skills = grouped[priority];
    if (skills.length === 0) continue;

    const label = priority.toUpperCase();
    lines.push(`[${label}]`);

    for (const match of skills) {
      const typeIcon = {
        domain: '[D]',
        runbook: '[R]',
        reference: '[REF]'
      }[match.skill.type];

      lines.push(`  ${typeIcon} ${match.name}: ${match.skill.description}`);
      lines.push(`      File: ${match.skill.file}`);

      if (match.matchedKeywords.length > 0) {
        lines.push(`      Matched: ${match.matchedKeywords.join(', ')}`);
      }
    }
    lines.push('');
  }

  lines.push('---');
  lines.push('Read referenced files before responding.');
  lines.push('');

  return lines.join('\n');
}

async function main() {
  // Read prompt from stdin
  let input = '';

  for await (const chunk of process.stdin) {
    input += chunk;
  }

  if (!input.trim()) {
    process.exit(0);
  }

  let promptData: PromptInput;
  try {
    promptData = JSON.parse(input);
  } catch {
    // Not valid JSON, use raw input as prompt
    promptData = { prompt: input };
  }

  const prompt = promptData.prompt || '';

  if (!prompt.trim()) {
    process.exit(0);
  }

  try {
    const rules = loadSkillRules();
    const matches = matchSkills(prompt, rules);
    const output = formatOutput(matches);

    if (output) {
      console.log(output);
    }
  } catch (error) {
    // Silently fail - don't block the user
    process.exit(0);
  }
}

main();
