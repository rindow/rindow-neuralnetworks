<?php
namespace Rindow\NeuralNetworks\Support\HDA;

use RuntimeException;
use PDO;
use PDOStatement;
use Iterator;

/**
 * @implements Iterator<mixed,mixed> 
 */
class HDASqliteIterator implements Iterator
{
    protected mixed $key;
    protected mixed $value;
    protected PDOStatement $stat;
    protected mixed $eof = false;
    protected bool $fetched = false;

    public function __construct(PDO $pdo, string $table, string $ancestor)
    {
        $stat = $pdo->prepare("SELECT * FROM ".$table." WHERE ancestor = :ancestor");
        if($stat==false) {
            throw new RuntimeException('PDO prepare error');
        }
        $this->stat = $stat;
        if(!$this->stat->execute([':ancestor'=>$ancestor]))
            throw new RuntimeException('query error in '.$ancestor);
    }

    protected function fetch() : void
    {
        if($this->fetched)
            return;
        if($this->eof)
            return;
        $row = $this->stat->fetch(PDO::FETCH_ASSOC);
        $this->fetched = true;
        if(!$row) {
            $this->eof = true;
            $this->value = null;
            $this->key = null;
            return;
        }
        $this->value = $row['value'];
        $this->key = $row['key'];
    }

    public function rewind() : void
    {
        $this->fetch();
    }

    public function valid() : bool
    {
        $this->fetch();
        return !$this->eof;
    }

    public function current() : mixed
    {
        $this->fetch();
        return $this->value;
    }

    public function key() : mixed
    {
        $this->fetch();
        return $this->key;
    }

    public function next() : void
    {
        $this->fetched = false;
        $this->fetch();
    }

    public function __destruct()
    {
        $this->stat->closeCursor();
    }
}
