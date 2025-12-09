"""
PQ Governance Engine MVP

Minimal viable implementation of proposal → review → vote → activation lifecycle.

Author: Manus-H
"""

import json
import time
import hashlib
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path


class ProposalStatus(Enum):
    """Status of a governance proposal."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED_FOR_VOTE = "approved_for_vote"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    ACTIVATED = "activated"
    CANCELLED = "cancelled"


class VoteChoice(Enum):
    """Vote choices."""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class Proposal:
    """
    PQ migration governance proposal.
    
    Attributes:
        proposal_id: Unique proposal identifier
        title: Human-readable title
        description: Detailed description
        epoch_start_block: Block number where new epoch starts
        algorithm_id: Hash algorithm ID for new epoch
        algorithm_name: Human-readable algorithm name
        rule_version: Consensus rule version
        proposer: Address/ID of proposer
        created_at: Unix timestamp of creation
        status: Current proposal status
        review_approvals: Number of review approvals
        votes_yes: Number of yes votes
        votes_no: Number of no votes
        votes_abstain: Number of abstain votes
        total_voting_power: Total voting power in system
    """
    
    proposal_id: str
    title: str
    description: str
    epoch_start_block: int
    algorithm_id: int
    algorithm_name: str
    rule_version: str
    proposer: str
    created_at: float
    status: ProposalStatus
    review_approvals: int = 0
    votes_yes: int = 0
    votes_no: int = 0
    votes_abstain: int = 0
    total_voting_power: int = 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def compute_hash(self) -> str:
        """Compute proposal hash for attestation."""
        # Hash proposal content (excluding mutable fields like status, votes)
        content = {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "epoch_start_block": self.epoch_start_block,
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "rule_version": self.rule_version,
            "proposer": self.proposer,
            "created_at": self.created_at,
        }
        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()


@dataclass
class Review:
    """Proposal review."""
    reviewer: str
    proposal_id: str
    approved: bool
    comment: str
    timestamp: float


@dataclass
class Vote:
    """Proposal vote."""
    voter: str
    proposal_id: str
    choice: VoteChoice
    voting_power: int
    timestamp: float


class GovernanceEngine:
    """
    Minimal governance engine for PQ migration.
    
    Implements proposal → review → vote → activation lifecycle.
    """
    
    def __init__(self, storage_dir: str = "./artifacts/governance"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.proposals: Dict[str, Proposal] = {}
        self.reviews: Dict[str, List[Review]] = {}
        self.votes: Dict[str, List[Vote]] = {}
        
        # Configuration
        self.min_review_approvals = 3
        self.review_approval_threshold = 0.667  # 2/3
        self.vote_quorum = 0.40  # 40%
        self.vote_approval_threshold = 0.667  # 66.7%
        
        # Load existing state
        self.load_state()
    
    def submit_proposal(
        self,
        title: str,
        description: str,
        epoch_start_block: int,
        algorithm_id: int,
        algorithm_name: str,
        rule_version: str,
        proposer: str,
    ) -> Proposal:
        """
        Submit a new PQ migration proposal.
        
        Args:
            title: Proposal title
            description: Detailed description
            epoch_start_block: Block number for epoch activation
            algorithm_id: Hash algorithm ID
            algorithm_name: Algorithm name
            rule_version: Consensus rule version
            proposer: Proposer identifier
            
        Returns:
            Created proposal
        """
        # Generate proposal ID
        proposal_id = f"pq-{int(time.time())}-{len(self.proposals)}"
        
        proposal = Proposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            epoch_start_block=epoch_start_block,
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_name,
            rule_version=rule_version,
            proposer=proposer,
            created_at=time.time(),
            status=ProposalStatus.SUBMITTED,
        )
        
        self.proposals[proposal_id] = proposal
        self.reviews[proposal_id] = []
        self.votes[proposal_id] = []
        
        self.save_state()
        
        print(f"Proposal {proposal_id} submitted")
        print(f"  Title: {title}")
        print(f"  Epoch start block: {epoch_start_block}")
        print(f"  Algorithm: {algorithm_name} (0x{algorithm_id:02x})")
        
        return proposal
    
    def submit_review(
        self,
        proposal_id: str,
        reviewer: str,
        approved: bool,
        comment: str,
    ) -> Review:
        """
        Submit a review for a proposal.
        
        Args:
            proposal_id: Proposal to review
            reviewer: Reviewer identifier
            approved: Whether reviewer approves
            comment: Review comment
            
        Returns:
            Created review
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status not in [ProposalStatus.SUBMITTED, ProposalStatus.UNDER_REVIEW]:
            raise ValueError(f"Proposal {proposal_id} is not under review")
        
        # Create review
        review = Review(
            reviewer=reviewer,
            proposal_id=proposal_id,
            approved=approved,
            comment=comment,
            timestamp=time.time(),
        )
        
        self.reviews[proposal_id].append(review)
        
        # Update proposal status
        if proposal.status == ProposalStatus.SUBMITTED:
            proposal.status = ProposalStatus.UNDER_REVIEW
        
        # Count approvals
        approvals = sum(1 for r in self.reviews[proposal_id] if r.approved)
        proposal.review_approvals = approvals
        
        # Check if enough reviews
        total_reviews = len(self.reviews[proposal_id])
        if total_reviews >= self.min_review_approvals:
            approval_rate = approvals / total_reviews
            if approval_rate >= self.review_approval_threshold:
                proposal.status = ProposalStatus.APPROVED_FOR_VOTE
                print(f"Proposal {proposal_id} approved for voting")
            elif approval_rate < 0.5:
                proposal.status = ProposalStatus.REJECTED
                print(f"Proposal {proposal_id} rejected by reviewers")
        
        self.save_state()
        
        return review
    
    def start_voting(self, proposal_id: str) -> None:
        """
        Start voting period for a proposal.
        
        Args:
            proposal_id: Proposal to start voting on
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.APPROVED_FOR_VOTE:
            raise ValueError(f"Proposal {proposal_id} not approved for voting")
        
        proposal.status = ProposalStatus.VOTING
        
        self.save_state()
        
        print(f"Voting started for proposal {proposal_id}")
    
    def submit_vote(
        self,
        proposal_id: str,
        voter: str,
        choice: VoteChoice,
        voting_power: int = 1,
    ) -> Vote:
        """
        Submit a vote on a proposal.
        
        Args:
            proposal_id: Proposal to vote on
            voter: Voter identifier
            choice: Vote choice
            voting_power: Voting power of voter
            
        Returns:
            Created vote
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.VOTING:
            raise ValueError(f"Proposal {proposal_id} is not in voting period")
        
        # Check if voter already voted
        for existing_vote in self.votes[proposal_id]:
            if existing_vote.voter == voter:
                raise ValueError(f"Voter {voter} already voted on {proposal_id}")
        
        # Create vote
        vote = Vote(
            voter=voter,
            proposal_id=proposal_id,
            choice=choice,
            voting_power=voting_power,
            timestamp=time.time(),
        )
        
        self.votes[proposal_id].append(vote)
        
        # Update vote counts
        if choice == VoteChoice.YES:
            proposal.votes_yes += voting_power
        elif choice == VoteChoice.NO:
            proposal.votes_no += voting_power
        elif choice == VoteChoice.ABSTAIN:
            proposal.votes_abstain += voting_power
        
        self.save_state()
        
        return vote
    
    def finalize_vote(self, proposal_id: str) -> bool:
        """
        Finalize voting and determine outcome.
        
        Args:
            proposal_id: Proposal to finalize
            
        Returns:
            True if proposal passed, False otherwise
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.VOTING:
            raise ValueError(f"Proposal {proposal_id} is not in voting period")
        
        # Calculate vote statistics
        total_votes = proposal.votes_yes + proposal.votes_no + proposal.votes_abstain
        turnout = total_votes / proposal.total_voting_power
        
        # Check quorum
        if turnout < self.vote_quorum:
            proposal.status = ProposalStatus.REJECTED
            print(f"Proposal {proposal_id} rejected: quorum not met ({turnout:.1%} < {self.vote_quorum:.1%})")
            self.save_state()
            return False
        
        # Calculate approval rate (excluding abstentions)
        votes_for_or_against = proposal.votes_yes + proposal.votes_no
        if votes_for_or_against == 0:
            approval_rate = 0.0
        else:
            approval_rate = proposal.votes_yes / votes_for_or_against
        
        # Check approval threshold
        if approval_rate >= self.vote_approval_threshold:
            proposal.status = ProposalStatus.PASSED
            print(f"Proposal {proposal_id} passed ({approval_rate:.1%} approval)")
            self.save_state()
            return True
        else:
            proposal.status = ProposalStatus.REJECTED
            print(f"Proposal {proposal_id} rejected ({approval_rate:.1%} < {self.vote_approval_threshold:.1%})")
            self.save_state()
            return False
    
    def activate_proposal(self, proposal_id: str) -> str:
        """
        Activate a passed proposal.
        
        Args:
            proposal_id: Proposal to activate
            
        Returns:
            Governance hash for epoch registration
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.PASSED:
            raise ValueError(f"Proposal {proposal_id} has not passed voting")
        
        # Compute governance hash
        governance_hash = "0x" + proposal.compute_hash()
        
        # Update status
        proposal.status = ProposalStatus.ACTIVATED
        
        self.save_state()
        
        print(f"Proposal {proposal_id} activated")
        print(f"  Governance hash: {governance_hash}")
        print(f"  Epoch start block: {proposal.epoch_start_block}")
        
        return governance_hash
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)
    
    def list_proposals(self, status: Optional[ProposalStatus] = None) -> List[Proposal]:
        """
        List all proposals, optionally filtered by status.
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of proposals
        """
        proposals = list(self.proposals.values())
        
        if status is not None:
            proposals = [p for p in proposals if p.status == status]
        
        return proposals
    
    def save_state(self) -> None:
        """Save governance state to disk."""
        # Save proposals
        proposals_file = self.storage_dir / "proposals.json"
        with open(proposals_file, 'w') as f:
            proposals_data = {pid: p.to_dict() for pid, p in self.proposals.items()}
            json.dump(proposals_data, f, indent=2)
        
        # Save reviews
        reviews_file = self.storage_dir / "reviews.json"
        with open(reviews_file, 'w') as f:
            reviews_data = {pid: [asdict(r) for r in reviews] for pid, reviews in self.reviews.items()}
            json.dump(reviews_data, f, indent=2)
        
        # Save votes
        votes_file = self.storage_dir / "votes.json"
        with open(votes_file, 'w') as f:
            votes_data = {}
            for pid, votes in self.votes.items():
                votes_data[pid] = []
                for v in votes:
                    vote_dict = asdict(v)
                    vote_dict["choice"] = v.choice.value
                    votes_data[pid].append(vote_dict)
            json.dump(votes_data, f, indent=2)
    
    def load_state(self) -> None:
        """Load governance state from disk."""
        # Load proposals
        proposals_file = self.storage_dir / "proposals.json"
        if proposals_file.exists():
            with open(proposals_file, 'r') as f:
                proposals_data = json.load(f)
                for pid, p_dict in proposals_data.items():
                    p_dict["status"] = ProposalStatus(p_dict["status"])
                    self.proposals[pid] = Proposal(**p_dict)
        
        # Load reviews
        reviews_file = self.storage_dir / "reviews.json"
        if reviews_file.exists():
            with open(reviews_file, 'r') as f:
                reviews_data = json.load(f)
                for pid, reviews_list in reviews_data.items():
                    self.reviews[pid] = [Review(**r) for r in reviews_list]
        
        # Load votes
        votes_file = self.storage_dir / "votes.json"
        if votes_file.exists():
            with open(votes_file, 'r') as f:
                votes_data = json.load(f)
                for pid, votes_list in votes_data.items():
                    self.votes[pid] = []
                    for v_dict in votes_list:
                        v_dict["choice"] = VoteChoice(v_dict["choice"])
                        self.votes[pid].append(Vote(**v_dict))


if __name__ == "__main__":
    # Example usage
    engine = GovernanceEngine()
    
    # Submit proposal
    proposal = engine.submit_proposal(
        title="Activate SHA3-256 for PQ Migration Phase 2",
        description="Transition to SHA3-256 as the post-quantum hash algorithm",
        epoch_start_block=10000,
        algorithm_id=0x01,
        algorithm_name="SHA3-256",
        rule_version="v2-dual-required",
        proposer="alice",
    )
    
    print(f"\nProposal created: {proposal.proposal_id}")
    
    # Submit reviews
    engine.submit_review(proposal.proposal_id, "reviewer1", True, "Looks good")
    engine.submit_review(proposal.proposal_id, "reviewer2", True, "Approved")
    engine.submit_review(proposal.proposal_id, "reviewer3", True, "LGTM")
    
    print(f"\nProposal status: {proposal.status.value}")
    
    # Start voting
    engine.start_voting(proposal.proposal_id)
    
    # Submit votes
    engine.submit_vote(proposal.proposal_id, "voter1", VoteChoice.YES, voting_power=30)
    engine.submit_vote(proposal.proposal_id, "voter2", VoteChoice.YES, voting_power=25)
    engine.submit_vote(proposal.proposal_id, "voter3", VoteChoice.NO, voting_power=10)
    engine.submit_vote(proposal.proposal_id, "voter4", VoteChoice.YES, voting_power=15)
    
    # Finalize vote
    passed = engine.finalize_vote(proposal.proposal_id)
    
    if passed:
        # Activate proposal
        governance_hash = engine.activate_proposal(proposal.proposal_id)
        print(f"\nProposal activated with governance hash: {governance_hash}")
